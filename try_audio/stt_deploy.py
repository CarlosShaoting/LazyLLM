import lazyllm

stt = lazyllm.TrainableModule('SenseVoiceSmall')

lazyllm.WebModule(stt, port=range(23466, 23470)).start().wait()