import lazyllm

from lazyllm.tools.rag import DocNode, post_process_video_audio_for_llm
from lazyllm.tools.rag.readers import AudioReader, VideoAudioReader

audio_path = "/home/mnt/cuishaoting/LazyLLM/try_audio/audio/sample-speech-1m.mp3"


# model = lazyllm.UrlModule(url="http://10.119.27.175:34451/generate")
# reader = AudioReader(model=model)
# nodes = reader._load_data(audio_path)
# print(nodes[0].text if hasattr(nodes[0], "text") else nodes[0].get_content())

# Whisper
# reader = VideoAudioReader()
# print('reader loaded')
# nodes = reader._load_data(audio_path, time_segment = True)
# print(nodes[0].text if hasattr(nodes[0], "text") else nodes[0].get_content())
# for n in nodes:
#     print(n.text)
#     print(n.metadata)

# test_post_process
def test_post_process():
    file_path = "/home/mnt/cuishaoting/LazyLLM/try_audio/audio/sample-speech-1m.mp3"
    nodes = [
        DocNode(
            text="Welcome to samplolid.com, a free online resource for downloading sample files in a wide variety",
            metadata={"start_time": 0.0, "end_time": 5.74, "file_path": file_path},
        ),
        # DocNode(
        #     text="of digital formats, whether you are a software developer testing file upload functionality,",
        #     metadata={"start_time": 5.74, "end_time": 11.1, "file_path": file_path},
        # ),
        # DocNode(
        #     text="a quality assurance engineer validating media players, a student learning about digital",
        #     metadata={"start_time": 11.52, "end_time": 16.36, "file_path": file_path},
        # ),
        # DocNode(
        #     text="formats, or simply someone who needs a quick test file, samplolid provides ready-to-use",
        #     metadata={"start_time": 16.36, "end_time": 22.0, "file_path": file_path},
        # ),
    ]

    processed_nodes = post_process_video_audio_for_llm(nodes, output_dir = '/home/mnt/cuishaoting/LazyLLM/try_audio/audio/video_seg.mp3')
    for node in processed_nodes:
        print(node.text)
        print(node.metadata)


if __name__ == "__main__":
    # test_post_process()
    pass

from lazyllm.tools.rag import ImageRetriever

retriever = ImageRetriever(
    doc=documents,
    group_name="your_group",
    topk=3,
)

results = retriever("/path/to/query.jpg")
