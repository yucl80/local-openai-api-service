from llama_cpp.server.types import (
    CreateCompletionRequest,
    CreateEmbeddingRequest,
    CreateChatCompletionRequest,
)
from llama_cpp.llama_types import (
    ChatCompletionStreamResponseChoice,
    ChatCompletionStreamResponseDelta,
    ChatCompletionMessageToolCallChunk,
    ChatCompletionMessageToolCallChunkFunction,
    CreateEmbeddingResponse,
    Embedding,
    EmbeddingUsage,
)
import json
import uuid
import ast
import llama_cpp
import time
from optimum.onnxruntime import ORTModelForCustomTasks
from transformers import AutoTokenizer

from typing import (
    List,
    Optional,
    Union,
)


def process_ast_node(node):
    # Check if the node is a function call
    if isinstance(node, ast.Call):
        # Return a string representation of the function call
        return ast.unparse(node)
    else:
        # Convert the node to source code and evaluate to get the value
        node_str = ast.unparse(node)
        return eval(node_str)


def parse_python_function_call(call_str):
    tree = ast.parse(call_str)
    expr = tree.body[0]

    call_node = expr.value
    function_name = (
        call_node.func.id
        if isinstance(call_node.func, ast.Name)
        else str(call_node.func)
    )

    parameters = {}
    noNameParam = []

    # Process positional arguments
    for arg in call_node.args:
        noNameParam.append(process_ast_node(arg))

    # Process keyword arguments
    for kw in call_node.keywords:
        parameters[kw.arg] = process_ast_node(kw.value)

    if noNameParam:
        parameters["None"] = noNameParam

    function_dict = {"name": function_name, "arguments": parameters}
    return function_dict


FN_CALL_DELIMITER = "<<function>>"


def strip_function_calls(content: str) -> list[str]:
    """
    Split the content by the function call delimiter and remove empty strings
    """
    return [
        element.strip()
        for element in content.split(FN_CALL_DELIMITER)[1:]
        if element.strip()
    ]


def parse_function_call(call: str) -> dict[str, any]:
    """
    This is temporary. The long term solution is to union all the
    types of the parameters from the user's input function definition,
    and check which language is a proper super set of the union type.
    """
    try:
        return parse_python_function_call(call)
    except Exception as e:
        # If Python parsing fails, try Java parsing

        return None


def get_openfunctions_prompt(messages: list = [], functions: list = []) -> str:
    """
    Generates a conversation prompt based on the user's query and a list of functions.

    Parameters:
    - user_query (str): The user's query.
    - functions (list): A list of functions to include in the prompt.

    Returns:
    - str: The formatted conversation prompt.
    """
    system = "You are an AI programming assistant"  # , utilizing the Gorilla LLM model, developed by Gorilla LLM, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer."
    if len(messages) > 0:
        if messages[0]["role"] == "system":
            system = messages[0]["content"]
            messages = messages[1:]

    user_query = ""
    for message in messages:
        if message["role"] == "user":
            user_query += "### Instruction: <<question>> " + message["content"] + "\n"
        elif message["role"] == "system":
            user_query += message["content"]
        else:
            user_query += "### Response:\n" + message["content"] + "\n<|EOT|>\n"

    if len(functions) == 0:
        return f"{system}\n### Instruction: <<question>> {user_query}\n### Response: "
    functions_string = json.dumps(functions)
    result = f"<｜begin▁of▁sentence｜>{system}\n### Instruction: <<function>>{functions_string}\n{user_query} ### Response: "

    print(result)
    return result


def format_response(response):
    result = []
    choices = response["choices"]
    for choice in choices:
        text = choice["text"]
        calls = strip_function_calls(text)
        tool_calls = []
        for function_call in calls:
            fc = parse_function_call(function_call)
            if fc is not None:
                tool_calls.append(
                    {
                        "id": "call_" + uuid.uuid4().hex,
                        "function": {
                            "name": fc["name"],
                            "arguments": json.dumps(fc["arguments"]),
                        },
                        "type": "function",
                    }
                )
        result.append(
            {
                "finish_reason": choice["finish_reason"],
                "index": choice["index"],
                "logprobs": choice["logprobs"],
                "message": {
                    "content": text,
                    "role": "assistant",
                    "tool_calls": tool_calls,
                },
            }
        )
    return result


def handle_openfunction(body: CreateCompletionRequest, llama) -> any:
    tools = body.tools
    if tools is None:
        tools = []
    user_prompt = get_openfunctions_prompt(body.messages, tools)
    response = llama(
        user_prompt,
        stop=["<|EOT|>"],
        temperature=body.temperature,
        top_p=body.top_p,
        max_tokens=body.max_tokens,
    )
    tool_calls = format_response(response)
    return {
        "id": response["id"],
        "object": "chat.completion",
        "created": response["created"],
        "model": body.model,
        "choices": tool_calls,
        "usage": {
            "prompt_tokens": response["usage"]["prompt_tokens"],
            "completion_tokens": response["usage"]["completion_tokens"],
            "total_tokens": response["usage"]["total_tokens"],
        },
    }


def openfunction_stream_chat(body: CreateCompletionRequest, llama) -> any:
    tools = body.tools
    if tools is None:
        tools = []
    user_prompt = get_openfunctions_prompt(body.messages, tools)
    if tools is None:
        for chunk in llama(
            user_prompt,
            stop=["<|EOT|>"],
            temperature=body.temperature,
            top_p=body.top_p,
            max_tokens=body.max_tokens,
            stream=True,
        ):
            choices = [
                ChatCompletionStreamResponseChoice(
                    index=1,
                    delta=ChatCompletionStreamResponseDelta(
                        content=choice["text"],
                    ),
                    finish_reason=choice["finish_reason"],
                    logprobs=choice["logprobs"],
                )
                for choice in chunk["choices"]
            ]
            chatCompletionChunk = llama_cpp.ChatCompletionChunk(
                id="chatcmpl-" + uuid.uuid4().hex,
                model=body.model,
                object="chat.completion.chunk",
                created=int(time.time()),
                choices=choices,
            )
            yield chatCompletionChunk
    else:
        response = llama(
            user_prompt,
            stop=["<|EOT|>"],
            temperature=body.temperature,
            top_p=body.top_p,
            max_tokens=body.max_tokens,
            stream=False,
        )
        tool_calls = format_response(response)
        choices = response["choices"]
        stream_response_choices = []
        choices_index = 0
        for choice in choices:
            text = choice["text"]
            calls = strip_function_calls(text)
            tool_calls = []
            index = 0
            for function_call in calls:
                fc = parse_function_call(function_call)
                if fc is not None:
                    tool_calls.append(
                        ChatCompletionMessageToolCallChunk(
                            index=index,
                            id="call_" + uuid.uuid4().hex,
                            type="function",
                            function=ChatCompletionMessageToolCallChunkFunction(
                                name=fc["name"],
                                arguments=json.dumps(fc["arguments"]),
                            ),
                        )
                    )
                index += 1
            stream_response_choices.append(
                ChatCompletionStreamResponseChoice(
                    index=choices_index,
                    delta=ChatCompletionStreamResponseDelta(
                        content=json.dumps(calls),
                        finish_reason=choice["finish_reason"],
                        logprobs=choice["logprobs"],
                        tool_calls=tool_calls,
                    ),
                )
            )
            choices_index += 1

        chatCompletionChunk = llama_cpp.ChatCompletionChunk(
            id="chatcmpl-" + uuid.uuid4().hex,
            model=body.model,
            object="chat.completion.chunk",
            created=int(time.time()),
            choices=stream_response_choices,
        )
        yield chatCompletionChunk


def functionary_stream_chat(body: CreateCompletionRequest, llama) -> any:
    response = llama.create_chat_completion(
        messages=body.messages, tools=body.tools, tool_choice="auto", stream=False
    )
    stream_response_choices = []
    choices = response["choices"]
    choices_index = 0
    for choice in choices:
        tool_calls = []
        calls = choice["message"]["tool_calls"]
        index = 0
        for call in calls:
            tool_calls.append(
                ChatCompletionMessageToolCallChunk(
                    index=index,
                    id=call["id"],
                    type=call["type"],
                    function=ChatCompletionMessageToolCallChunkFunction(
                        name=call["function"]["name"],
                        arguments=json.dumps(call["function"]["arguments"]),
                    ),
                )
            )
            index += 1
        stream_response_choices.append(
            ChatCompletionStreamResponseChoice(
                index=choices_index,
                delta=ChatCompletionStreamResponseDelta(
                    content=None,
                    finish_reason=choice["finish_reason"],
                    logprobs=choice["logprobs"],
                    tool_calls=tool_calls,
                ),
            )
        )
        choices_index += 1

    chatCompletionChunk = llama_cpp.ChatCompletionChunk(
        id=response["id"],
        model=body.model,
        object="chat.completion.chunk",
        created=response["created"],
        choices=stream_response_choices,
    )
    print(chatCompletionChunk)
    yield chatCompletionChunk


def handle_firefunction(body: CreateChatCompletionRequest, llama) -> any:
    messages = []
    function_spec = json.dumps(body.tools)
    messages.append({"role": "functions", "content": function_spec})
    for message in body.messages:
        messages.append({"role": message["role"], "content": message["content"]})
    response = llama.create_chat_completion(
        messages=messages,
        tools=function_spec,
        tool_choice="auto",
        temperature=body.temperature,
        top_p=body.top_p,
        logprobs=body.logprobs,
        max_tokens=body.max_tokens,
    )
    choices = []
    for choice in response["choices"]:
        message_content = choice["message"]["content"]
        if "<functioncall>" in message_content:
            function_call_json = message_content[len("<functioncall>") :]
            function_call_data = json.loads(function_call_json)
            choices.append(
                {
                    "index": choice["index"],
                    "logprobs": choice["logprobs"],
                    "finish_reason": choice["finish_reason"],
                    "message": {
                        "content": message_content,
                        "role": choice["message"]["role"],
                        "tool_calls": [
                            {
                                "id": "tool_call_" + uuid.uuid4().hex,
                                "type": "function",
                                "function": {
                                    "name": function_call_data["name"],
                                    "arguments": json.dumps(
                                        function_call_data["arguments"]
                                    ),
                                },
                            }
                        ],
                    },
                }
            )
        else:
            choices.append(
                {
                    "index": choice["index"],
                    "logprobs": choice["logprobs"],
                    "finish_reason": choice["finish_reason"],
                    "message": {
                        "content": message_content,
                        "role": choice["message"]["role"],
                    },
                }
            )

    result = {
        "id": response["id"],
        "object": response["object"],
        "created": response["created"],
        "model": body.model,
        "choices": choices,
        "usage": {
            "prompt_tokens": response["usage"]["prompt_tokens"],
            "completion_tokens": response["usage"]["completion_tokens"],
            "total_tokens": response["usage"]["total_tokens"],
        },
    }
    return result


class BgeOnnxModel:
    _model: str
    _tokenizer: any
    __ORTModel: ORTModelForCustomTasks

    def __init__(self, model_path: str, model_alias: str, **kwargs: any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)
        self._model = model_alias
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._ORTModel = ORTModelForCustomTasks.from_pretrained(model_path)

    def create_embedding(
        self, input: Union[str, List[str]], model: Optional[str] = None
    ) -> CreateEmbeddingResponse:
        """Embed a string.

        Args:
            input: The utf-8 encoded string to embed.

        Returns:
            An embedding object.
        """

        input = input if isinstance(input, list) else [input]

        # get numeric embeddings
        embeds: Union[List[List[float]], List[List[List[float]]]]

        inputs_token = self._tokenizer(
            input,
            padding="longest",
            return_tensors="np",
        )
        embeds = self._ORTModel.forward(**inputs_token)["sentence_embedding"].tolist()
        total_tokens = len(inputs_token["input_ids"][0])

        # convert to CreateEmbeddingResponse
        data: List[Embedding] = [
            {
                "object": "embedding",
                "embedding": emb,
                "index": idx,
            }
            for idx, emb in enumerate(embeds)
        ]

        return CreateEmbeddingResponse(
            object="list",
            model=self._model,
            data=data,
            usage=EmbeddingUsage(
                prompt_tokens=total_tokens,
                total_tokens=total_tokens,
            ),
        )
