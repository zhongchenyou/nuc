import pyaudio
import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat, ResultCallback
from openai import OpenAI
import time

# DashScope API 配置
dashscope.api_key = "sk-0536c955482c4eba8458c059ddd8ce8d"

# OpenAI API 配置
openai_api_key = "sk-0536c955482c4eba8458c059ddd8ce8d"
openai_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
openai_client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)

# 语音识别回调类
class RecognitionCallbackClass(RecognitionCallback):
    def __init__(self):
        self.current_sentence = ""  # 当前完整句子缓存
        self.sentence_ready = False  # 是否完成一句话标志

    def on_event(self, result: RecognitionResult) -> None:
        try:
            if "output" in result and "sentence" in result["output"]:
                sentence = result["output"]["sentence"]
                text = sentence.get("text", "（无文本）")
                self.current_sentence = text

                # 如果句子结束，设置标志
                if sentence.get("sentence_end", False):
                    self.sentence_ready = True
        except Exception as e:
            print(f"处理识别结果时发生错误: {e}")

# 语音合成回调类
class TTSCallback(ResultCallback):
    _player = None
    _stream = None
    is_connected = False  # 新增连接状态标志

    def on_open(self):
        print("语音合成连接已建立。")
        self._player = pyaudio.PyAudio()
        self._stream = self._player.open(
            format=pyaudio.paInt16, channels=1, rate=22050, output=True
        )

    def on_complete(self):
        print("语音合成任务成功完成。")

    def on_error(self, message: str):
        print(f"语音合成任务失败，错误信息: {message}")

    def on_close(self):
        print("语音合成连接已关闭。")
        # 停止播放器
        self._stream.stop_stream()
        self._stream.close()
        self._player.terminate()

    def on_event(self, message):
        print(f"接收到事件消息: {message}")

    def on_data(self, data: bytes) -> None:
        print("接收到音频数据块，长度:", len(data))
        self._stream.write(data)

# 配置语音合成器
model = "cosyvoice-v1"
voice = "longxiaochun"
callback = TTSCallback()
synthesizer = SpeechSynthesizer(
    model=model,
    voice=voice,
    format=AudioFormat.PCM_22050HZ_MONO_16BIT,
    callback=callback,
)

# 初始化语音识别
recognition = Recognition(
    model='paraformer-realtime-v2',
    format='pcm',
    sample_rate=16000,
    callback=RecognitionCallbackClass()
)

# PyAudio 初始化
audio = pyaudio.PyAudio()

# 获取 OpenAI 的回答
def get_completion(messages, response_format="text", model="qwen2.5-72b-instruct"):
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=150,
        temperature=0,
    )
    return response.choices[0].message.content

# 清理文本中的换行符
def remove_newlines(text):
    return text.replace("\n", " ").strip()

# 语音识别函数
def start_recognition():
    """开始录音和识别"""
    recognition.start()

    # 打开音频流
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=1024
    )

    print("开始录音并识别...")

    try:
        while True:
            # 读取音频流数据
            buffer = stream.read(1024, exception_on_overflow=False)
            recognition.send_audio_frame(buffer)

            # 检查是否完成一句话
            if recognition._callback.sentence_ready:
                print(f"完整识别结果: {recognition._callback.current_sentence}")
                recognition._callback.sentence_ready = False  # 重置标志
                return recognition._callback.current_sentence  # 返回识别的句子
    except KeyboardInterrupt:
        print("手动停止识别")
    finally:
        stream.stop_stream()
        stream.close()
        recognition.stop()

# instruction = """
# 你的任务是以名为小韵的人物形象回答用户关于唐诗三百首的问题，每次回答时尽量提供丰富的背景、解析或相关信息，让用户更深入地了解诗词和文化背景。
# 若不知道某个问题，委婉地表达你无法回答，不要胡乱猜测或提供不准确的信息。
# 要避免成为话题的终结者，尽可能引导用户进行进一步的探索或提问，使互动更加流畅有趣，要鼓励用户继续提问或引导他们深入思考诗歌的意境与含义。
# 确保你的回答准确且富有教育性，避免模棱两可或过于简略的回答。
# 回答时不要模棱两可，要清晰明确地解释每个问题，让用户获得充分的知识。
# """

instruction = """
你的任务是以名为小韵的人物形象回答用户关于唐诗三百首的问题，每次回答时言简意赅，不要啰嗦
若不知道某个问题，委婉地表达你无法回答，不要胡乱猜测或提供不准确的信息。  
要避免成为话题的终结者，尽可能引导用户进行进一步的探索或提问，使互动更加流畅有趣，要鼓励用户继续提问或引导他们深入思考诗歌的意境与含义。
确保你的回答准确且富有教育性
回答时不要模棱两可，要清晰明确地解释每个问题，让用户获得充分的知识。
"""

# 主程序循环
conversation_history = [{"role": "system", "content": instruction}]

print("欢迎使用语音助手——小韵！输入问题即可开始对话，输入“退出”结束程序。\n")

while True:
    # 用户输入语音
    recognized_text = start_recognition()

    # 退出条件
    if recognized_text.lower() in ["退出", "exit", "q", "quit"]:
        print("结束对话。")
        break

    # 添加用户输入到对话历史记录
    conversation_history.append({"role": "user", "content": recognized_text})

    # 获取 OpenAI 模型的回答
    response_text = get_completion(conversation_history)

    # 打印文本回复
    print("小韵:", response_text)

    # 使用语音合成播放回复
    try:
        cleaned_text = remove_newlines(response_text)
        print("正在生成语音...")

        if not callback.is_connected:
            print("连接已关闭，重新初始化...")

        synthesizer.streaming_call(cleaned_text)
        synthesizer.streaming_complete(complete_timeout_millis=30000)
    except Exception as e:
        print(f"语音合成失败: {e}")
        print("重新初始化合成器...")
        synthesizer = SpeechSynthesizer(
            model=model,
            voice=voice,
            format=AudioFormat.PCM_22050HZ_MONO_16BIT,
            callback=callback,
        )
        time.sleep(5)  # 等待资源释放
        synthesizer.streaming_call(cleaned_text)
        synthesizer.streaming_complete(complete_timeout_millis=30000)

print("程序结束")
