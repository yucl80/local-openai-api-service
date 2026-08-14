

# local-openai-api-service

#在llama-cpp-python的基础上增加了对ChatGLM3、GLM-4的支持，增加了对函数调用模型firefunction-v1，Gorilla OpenFunctions v2 的支持, bge-large-zh-v1.5, bge-m3 ,functionary-v2.5

## 运行说明

安装依赖后，运行 `python __main__.py` 即可启动服务。可通过 `MODEL` 环境变量指定模型路径，或使用 `server.cfg` 配置文件加载参数。
