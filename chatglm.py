import asyncio
import json
import logging
import time
import os
import uuid
import chatglm_cpp
import llama_cpp
from sse_starlette.sse import EventSourceResponse
from pprint import pprint
from fastapi import HTTPException, status
from llama_cpp.server.types import ChatCompletionRequestMessage
from llama_cpp.llama_types import (
    ChatCompletionResponseChoice,
    ChatCompletionMessageToolCall,
    ChatCompletionStreamResponseChoice,
    CreateChatCompletionStreamResponse,
    CompletionUsage,
    ChatCompletionStreamResponseDelta,
    CreateChatCompletionResponse,
    ChatCompletionResponseMessage,
    ChatCompletionRequestAssistantMessage,
    ChatCompletionMessageToolCallFunction,
    ChatCompletionStreamResponseDeltaEmpty,
)


def _buid_msg(body: ChatCompletionRequestMessage):
    messages = []
    if body.tools:
        system_content = (
            "Answer the following questions as best as you can. You have access to the following tools:\n"
            + json.dumps(body.tools, indent=4)
        )
        messages.insert(
            0, chatglm_cpp.ChatMessage(role="system", content=system_content)
        )
    for msg in body.messages:
        role = msg["role"]
        if role not in ["user", "assistant", "system", "observation"]:
            role = "observation"
        content = msg["content"]
        if isinstance(content, str):
            messages.append(chatglm_cpp.ChatMessage(role=role, content=content))
        else:
            for text in content:
                messages.append(chatglm_cpp.ChatMessage(role=role, content=text["text"] ))                            
       
    return messages


def stream_chat(
    chatglm_pipeline: chatglm_cpp.Pipeline,
    body: ChatCompletionRequestMessage,
    max_context_length: int,
    num_threads: int,
):
    max_tokens = 1024
    if body.max_tokens:
        max_tokens = body.max_tokens

    for chunk in chatglm_pipeline.chat(
        messages=_buid_msg(body),
        max_length=max_tokens,
        max_context_length=max_context_length,
        do_sample=body.temperature > 0,
        top_p=body.top_p,
        temperature=body.temperature,
        num_threads=num_threads,
        stream=True,
    ):
        choices = [
            ChatCompletionStreamResponseChoice(
                index=1,
                delta=ChatCompletionStreamResponseDelta(
                    content=chunk.content, role=chunk.role
                ),
                finish_reason=None,
                logprobs=None,
            )
        ]
        chunk = llama_cpp.ChatCompletionChunk(
            id="chatcmpl-" + uuid.uuid4().hex,
            model=body.model,
            object="chat.completion.chunk",
            created=int(time.time()),
            choices=choices,
        )
        yield chunk


def create_chat_completion(
    chatglm_pipeline: chatglm_cpp.Pipeline,
    body: ChatCompletionRequestMessage,
    max_context_length: int,
    num_threads: int,
) -> CreateChatCompletionResponse:
    def to_json_arguments(arguments):
        def tool_call(**kwargs):
            return kwargs

        try:
            return json.dumps(eval(arguments, dict(tool_call=tool_call)))
        except Exception:
            return arguments

    if not body.messages:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty messages")

    max_tokens = 2048
    if body.max_tokens:
        max_tokens = body.max_tokens

    messages = _buid_msg(body)

    output = chatglm_pipeline.chat(
        messages=messages,
        max_length=max_tokens,
        max_context_length=max_context_length,
        do_sample=body.temperature > 0,
        top_p=body.top_p,
        temperature=body.temperature,
        # num_threads=num_threads,
    )
    print("raw output: ", output)
    prompt_tokens = len(
        chatglm_pipeline.tokenizer.apply_chat_template(messages, max_context_length)
    )
    completion_tokens = len(
        chatglm_pipeline.tokenizer.encode(output.content, max_tokens)
    )

    finish_reason = "stop"
    tool_calls = None
    if output.tool_calls:
        tool_calls = [
            ChatCompletionMessageToolCall(
                id="tool_call_" + uuid.uuid4().hex,
                type=tool_call.type,
                function=ChatCompletionMessageToolCallFunction(
                    name=tool_call.function.name,
                    arguments=to_json_arguments(tool_call.function.arguments),
                ),
            )
            for tool_call in output.tool_calls
        ]
        finish_reason = "tool_calls"

    if tool_calls is None:
        choices = [
            ChatCompletionResponseChoice(
                index=0,
                message=ChatCompletionResponseMessage(
                    role="assistant", content=output.content
                ),
                finish_reason=finish_reason,
                logprobs=None,
            )
        ]
    else:
        choices = [
            ChatCompletionResponseChoice(
                index=0,
                message=ChatCompletionRequestAssistantMessage(
                    role="assistant", content=output.content, tool_calls=tool_calls
                ),
                finish_reason=finish_reason,
                logprobs=None,
            )
        ]

    response = CreateChatCompletionResponse(
        id="chatcmpl",
        object="chat.completion",
        created=int(time.time()),
        model="chatglm",
        choices=choices,
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
    print(response)
    return response
