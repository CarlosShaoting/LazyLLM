import lazyllm
from lazyllm.tools.rag import DocNode, post_process_video_audio_for_llm
from lazyllm.tools.rag.readers import ImageEmbReader, VideoAudioReader, VideoFrameReader

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
# def test_post_process():
#     file_path = "/home/mnt/cuishaoting/LazyLLM/try_audio/audio/sample-speech-1m.mp3"
#     nodes = [
#         DocNode(
#             text="Welcome to samplolid.com, a free online resource for downloading sample files in a wide variety",
#             metadata={"start_time": 0.0, "end_time": 5.74, "file_path": file_path},
#         ),
#         # DocNode(
#         #     text="of digital formats, whether you are a software developer testing file upload functionality,",
#         #     metadata={"start_time": 5.74, "end_time": 11.1, "file_path": file_path},
#         # ),
#         # DocNode(
#         #     text="a quality assurance engineer validating media players, a student learning about digital",
#         #     metadata={"start_time": 11.52, "end_time": 16.36, "file_path": file_path},
#         # ),
#         # DocNode(
#         #     text="formats, or simply someone who needs a quick test file, samplolid provides ready-to-use",
#         #     metadata={"start_time": 16.36, "end_time": 22.0, "file_path": file_path},
#         # ),
#     ]

#     processed_nodes = post_process_video_audio_for_llm(nodes)
#     for node in processed_nodes:
#         print(node.text)
#         print(node.metadata)


# if __name__ == "__main__":
#     # test_post_process()
#     pass

# from lazyllm.tools.rag import ImageRetriever

# retriever = ImageRetriever(
#     doc=documents,
#     group_name="your_group",
#     topk=3,
# )

# results = retriever("/path/to/query.jpg")


# # Whisper + time_segment and interval
# reader = VideoAudioReader(time_interval=15, time_segment=True)
# vp = '/home/mnt/cuishaoting/LazyLLM/try_audio/video/speech.mp4'
# print(f'{vp} loaded')
# nodes = reader._load_data(vp)
# for n in nodes:
#     print([n.text, n.metadata])
#     print('='*30)
# print("Processed Nodes")
# pns = post_process_video_audio_for_llm([nodes[0]], output_dir = '/home/mnt/cuishaoting/LazyLLM/try_audio/video')
# print(pns[0].text)
# print(pns[0].metadata)

# VideoFrameReader

# vp = '/home/mnt/cuishaoting/LazyLLM/try_audio/video/speech.mp4'
# raw_model = lazyllm.TrainableModule('siglip').start()
# model = _EmbedWrapper(raw_model)
# reader = VideoFrameReader(time=15, embed_model=model)
# print(f'{vp} loaded')
# nodes = reader._load_data(vp)
# for n in nodes:
#     embs = n.embedding["img_emb"][:10]
#     print([n.text, f'embedding 0 - 3 {embs}', n.metadata])
#     print('=' * 30)

# raw_model.stop()


def print_retrieved_nodes(nodes):
    for node in nodes:
        node_type = type(node).__name__
        content = node.image_path if hasattr(node, 'image_path') else node.text
        embedding = []
        if getattr(node, 'embedding', None):
            first_key = next(iter(node.embedding.keys()), None)
            if first_key:
                embedding = node.embedding[first_key][:3]
        print({
            'node_type': node_type,
            'content': content,
            'embedding_top3': embedding,
            'metadata': node.metadata,
        })
        print('=' * 30)
        return


if __name__ == '__main__':
    dataset_path = '/home/mnt/cuishaoting/LazyLLM/try_audio/test_content'
    # emb_model = lazyllm.UrlModule(url="http://10.119.16.229:39352/embeddings")
    emb_model = lazyllm.TrainableModule('siglip').start()
    # 用url报错
    #.deploy_method(
#         lazyllm.deploy.vllm,
#     url='http://10.119.16.229:39352/embeddings'
# )

    documents = lazyllm.Document(
        dataset_path=dataset_path,
        embed={'siglip': emb_model},
        manager=False,
        store_conf={'type': 'map'},
    )

    documents.add_reader('*.mp3', VideoAudioReader(time_segment=True))
    image_reader = ImageEmbReader()
    for ext in ('*.jpg'):
        documents.add_reader(ext, image_reader)
    documents.add_reader('*.mp4', VideoFrameReader(time=20))
    documents.activate_group('image', embed_keys=['siglip'])
    # documents.activate_group('lazyllm_root', embed_keys=['siglip'])
    documents.activate_group('lazyllm_root')

    retriever = lazyllm.Retriever(
        documents,
        group_name='image',
        similarity='cosine',
        topk=1,
        embed_keys=['siglip'],
    )
    retriever2 = lazyllm.Retriever(
        documents,
        group_name='lazyllm_root',
        similarity='bm25',
        topk=1,
    )

    print('Retriever image group, target: "狗"')
    nodes = retriever('狗')
    print_retrieved_nodes(nodes)
    print('Retriever root group, target: "Welcome to samploid.com"')
    nodes2 = retriever2('Welcome to samploid.com')
    print_retrieved_nodes(nodes2)
    emb_model.stop()

