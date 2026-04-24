# import lazyllm
# from lazyllm.tools.rag.readers import VideoAudioReader

# tts = remote_model = lazyllm.TrainableModule('SenseVoiceSmall').deploy_method(
#     lazyllm.deploy.vllm,
#     url='http://10.119.27.175:34451/generate',
    
# )
# print(tts('/home/mnt/cuishaoting/LazyLLM/try_audio/audio/sample-speech-1m.mp3'))

# # documents = lazyllm.Document(
# #     dataset_path="/home/mnt/cuishaoting/LazyLLM/try_audio/audio",
# #     embed=tts,
# # )

# # documents.add_reader("*.mp3", VideoAudioReader)
# # documents.create_node_group(
# #     name="audio_chunks",
# #     transform=lambda s: s.split('.'),
# # )

# # retriever = lazyllm.Retriever(doc=documents, group_name="audio_chunks", topk=3)
# # print(retriever("hi"))

# import base64
# import mimetypes
# import lazyllm

# audio_path = "/home/mnt/cuishaoting/LazyLLM/try_audio/audio/sample-speech-1m.mp3"
# mime_type, _ = mimetypes.guess_type(audio_path)

# with open(audio_path, "rb") as f:
#     audio_b64 = base64.b64encode(f.read()).decode("utf-8")

# stt = lazyllm.UrlModule(url="http://10.119.27.175:34451/generate")
# print(stt({
#     "inputs": "",
#     "audio": f"data:{mime_type};base64,{audio_b64}",
# }))