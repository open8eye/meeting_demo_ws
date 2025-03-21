"""
一个会议记录程序demo
由来：因为采用ws 流式收集音频 同时采用 fsmn-vad（语音端点检测模型）进行分段收集语音 模型无法区分具体的说人
解决方法：在fsmn-vad后的语音片段按照会话id进行存储 然后使用 3dspeaker 来进行说话人标记
"""
import os
import re
import sys
import time
import json
import uuid
import torch
import uvicorn
import argparse
import traceback
import numpy as np
from loguru import logger
from funasr import AutoModel
from urllib.parse import parse_qs
from pydantic import BaseModel, Field
from modelscope.pipelines import pipeline
from fastapi.responses import JSONResponse
from pydantic_settings import BaseSettings
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException

# 打开日志输出
logger.remove()
log_format = "{time:YYYY-MM-DD HH:mm:ss} [{level}] {file}:{line} - {message}"
logger.add(sys.stdout, format=log_format, level="DEBUG", filter=lambda record: record["level"].no < 40)
logger.add(sys.stderr, format=log_format, level="ERROR", filter=lambda record: record["level"].no >= 40)

"""
类 - start - ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
"""


class Config(BaseSettings):
    """应用程序配置类，包含各种参数设置。

    Attributes:
        sv_thr (float): Speaker verification阈值，用于判断是否匹配注册说话人。
        chunk_size_ms (int): 音频分块处理的毫秒数。
        sample_rate (int): 采样率（Hz）。
        bit_depth (int): 位深度。
        channels (int): 音频通道数。
        avg_logprob_thr (float): 平均log概率阈值。
    """
    # 可以自定义得分阈值来进行识别，阈值越高，判定为同一人的条件越严格
    # sv_thr: float = Field(0.4, description="说话人验证阈值")
    sv_thr: float = Field(0.31, description="说话人验证阈值")
    # sv_thr: float = Field(0.249, description="说话人验证阈值")
    # sv_thr: float = Field(0.2, description="说话人验证阈值")
    # sv_thr: float = Field(0.262, description="说话人验证阈值")
    chunk_size_ms: int = Field(300, description="分块大小（以毫秒为单位）")
    sample_rate: int = Field(16000, description="采样率（Hz）")
    bit_depth: int = Field(16, description="位深度")
    channels: int = Field(1, description="音频通道数")
    avg_logprob_thr: float = Field(-0.25, description="平均对数阈值")


class TranscriptionResponse(BaseModel):
    """API响应模型类。

    Attributes:
        code (int): 响应状态码。
        info (str): 附加信息或错误消息。
        data (str): 处理后的文本数据。
        spk_id (str): 说话人id。
        spk_set (str): 说话人集合。
    """
    code: int
    info: str
    data: str
    spk_id: str
    spk_set: str


"""
类 - end - ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
"""

"""
静态数据 - start - ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
"""
# 设置模型缓存路径
os.environ["MODELSCOPE_CACHE"] = "./"
# 生成配置
config = Config()
# 心情字典
emo_dict = {
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
}
# 事件字典
event_dict = {
    "<|BGM|>": "🎼",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😀",
    "<|Cry|>": "😭",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    "<|Cough|>": "😷",
}
# 表情字典
emoji_dict = {
    "<|nospeech|><|Event_UNK|>": "❓",
    "<|zh|>": "",
    "<|en|>": "",
    "<|yue|>": "",
    "<|ja|>": "",
    "<|ko|>": "",
    "<|nospeech|>": "",
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "",
    "<|BGM|>": "🎼",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😀",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
    "<|Cry|>": "😭",
    "<|EMO_UNKNOWN|>": "",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    "<|Cough|>": "😷",
    "<|Sing|>": "",
    "<|Speech_Noise|>": "",
    "<|withitn|>": "",
    "<|woitn|>": "",
    "<|GBG|>": "",
    "<|Event_UNK|>": "",
}
# 语言字典
lang_dict = {
    "<|zh|>": "<|lang|>",
    "<|en|>": "<|lang|>",
    "<|yue|>": "<|lang|>",
    "<|ja|>": "<|lang|>",
    "<|ko|>": "<|lang|>",
    "<|nospeech|>": "<|lang|>",
}
# 心情set
emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}
# 事件set
event_set = {"🎼", "👏", "😀", "😭", "🤧", "😷"}

# 语音识别模型配置信息
vad_model = "fsmn-vad"
vad_kwargs = {'max_single_segment_time': 30000}
punc_model = "ct-punc"
spk_model = "cam++"
disable_update = True
model_asr = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
model_uniasr = "iic/speech_UniASR-large_asr_2pass-zh-cn-16k-common-vocab8358-tensorflow1-offline"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
"""
静态数据 - end - ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
"""

"""
数据 - start - ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
"""
# 会话列表
session_list = {}
"""
数据 - end - ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
"""

"""
准备模型 - start - ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
"""
# sv_name = 'speech_eres2net_sv_zh-cn_16k-common'  # ERes2Net说话人确认-中文-通用-200k-Spkrs
# sv_name = 'speech_eres2net_large_sv_zh-cn_3dspeaker_16k' # 3d speaker 2024年模型
# sv_name = 'speech_res2net_sv_zh-cn_3dspeaker_16k'  # Res2Net说话人确认-中文-3D-Speaker-16k
sv_name = 'speech_campplus_sv_zh-cn_16k-common'  # CAM++说话人确认-中文-通用-200k-Spkrs
# 输入一段多人对话的音频，本模型可以自动的识别音频中的对话人数，并且对其进行区分，适合用于客服对话、会议讨论、采访等场景，该系统配合语音识别可进一步搭建多人对话的语音识别系统。
# sv_name = 'speech_campplus_speaker-diarization_common'  # CAM++说话人日志-对话场景角色区分-通用
# 说话校验
sv_pipeline = pipeline(
    task='speaker-verification',
    model=f'iic/{sv_name}',
    model_revision='v1.0.0',
    device=device,
    disable_update=disable_update
)
# speech_UniASR
# model_asr = AutoModel(
#     model=model_uniasr,
#     model_revision="v2.0.4",
#     disable_update=disable_update,
#     device=device,
# )

# asr 可嵌入热词 可识别说话人模型
model_asr = AutoModel(
    model=model_asr,
    model_revision="v2.0.4",
    # 因为是websocket流式调用 是一段段音频波形数据传入的 模型里的cam++ 就无法使用 所以屏蔽
    # vad_model=vad_model,
    # vad_kwargs=vad_kwargs,
    # trust_remote_code=True,
    # punc_model=punc_model,
    # spk_model=spk_model,
    # end
    disable_update=disable_update,
    device=device,
)

# 实时分析音频流，识别语音与非语音区域
model_vad = AutoModel(
    model="fsmn-vad",
    model_revision="v2.0.4",
    disable_pbar=True,
    max_end_silence_time=500,
    # speech_noise_thres=0.6,
    disable_update=True,
)
# 日志输出
logger.info(f"语音识别模型：{model_asr}\n"
            f"3d speaker模型：{sv_name}\n"
            f"使用设备：{device}")
"""
准备模型 - end - ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
"""

"""
方法 - start - ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
"""


def format_str(s: str) -> str:
    """替换字符串中的特定标记为对应的emoji表情。

    Args:
        s (str): 需要处理的原始字符串。

    Returns:
        str: 替换后的字符串。
    """
    for sptk in emoji_dict:
        s = s.replace(sptk, emoji_dict[sptk])
    return s


def format_str_v2(s: str) -> str:
    """合并事件和情绪标记，并整理表情符号。

    Args:
        s (str): 需要处理的原始字符串。

    Returns:
        str: 处理后的字符串。
    """
    sptk_dict = {}
    for sptk in emoji_dict:
        sptk_dict[sptk] = s.count(sptk)
        s = s.replace(sptk, "")
    emo = "<|NEUTRAL|>"
    for e in emo_dict:
        if sptk_dict[e] > sptk_dict[emo]:
            emo = e
    for e in event_dict:
        if sptk_dict[e] > 0:
            s = event_dict[e] + s
    s += emo_dict[emo]
    for emoji in emo_set.union(event_set):
        s = s.replace(" " + emoji, emoji)
        s = s.replace(emoji + " ", emoji)
    return s.strip()


def format_str_v3(s: str) -> str:
    """进一步处理语言标记和合并相似标记。

    Args:
        s (str): 需要处理的原始字符串。

    Returns:
        str: 最终格式化的字符串。
    """

    def get_emo(s_str):
        return s_str[-1] if s_str[-1] in emo_set else None

    def get_event(s_str):
        return s_str[0] if s_str[0] in event_set else None

    s = s.replace("<|nospeech|><|Event_UNK|>", "❓")
    for lang in lang_dict:
        s = s.replace(lang, "<|lang|>")
    s_list = [format_str_v2(s_i).strip(" ") for s_i in s.split("<|lang|>")]
    new_s = " " + s_list[0]
    cur_ent_event = get_event(new_s)
    for i in range(1, len(s_list)):
        if len(s_list[i]) == 0:
            continue
        if get_event(s_list[i]) == cur_ent_event and get_event(s_list[i]) != None:
            s_list[i] = s_list[i][1:]
        cur_ent_event = get_event(s_list[i])
        if get_emo(s_list[i]) != None and get_emo(s_list[i]) == get_emo(new_s):
            new_s = new_s[:-1]
        new_s += s_list[i].strip().lstrip()
    new_s = new_s.replace("The.", " ")
    return new_s.strip()


def contains_chinese_english_number(s: str) -> bool:
    """检查字符串是否包含中文、英文或数字。

    Args:
        s (str): 需要检查的字符串。

    Returns:
        bool: 是否包含目标字符。
    """
    return bool(re.search(r'[\u4e00-\u9fffA-Za-z0-9]', s))


def speaker_verify(audio: np.ndarray, audio1: np.ndarray, sv_thr: float) -> tuple:
    """执行说话人验证，判断音频是否匹配注册的说话人。

    Args:
        audio (np.ndarray): 输入音频数据。
        audio1 (np.ndarray): 输入音频数据。
        sv_thr (float): 验证阈值。

    Returns:
        tuple: (是否匹配, 匹配的说话人名称)
    """
    res_sv = sv_pipeline([audio, audio1], thr=sv_thr)
    # res_sv = sv_pipeline([audio, audio1])
    # 如果有先验信息，输入实际的说话人数，会得到更准确的预测结果
    # result = sv_pipeline(input_wav, oracle_num=2)
    logger.info(
        f"[speaker_verify] audio_len: {len(audio)};audio1_len: {len(audio1)}; sv_thr: {sv_thr}; result:{res_sv}")
    return res_sv["score"] >= sv_thr


def asr(audio: np.ndarray, lang: str, cache: dict, use_itn: bool = False) -> dict:
    """
    使用ASR模型对音频进行语音识别。
    Args:
        audio (np.ndarray): 输入音频数据。
        lang (str): 语言设置。
        cache (dict): 缓存参数。
        use_itn (bool, optional): 是否启用反规范化处理。默认False。

    Returns:
        dict: 识别结果字典。
    """
    start_time = time.time()
    result = model_asr.generate(
        input=audio,
        cache=cache,
        language=lang.strip(),
        use_itn=use_itn,
        batch_size_s=60,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.debug(f"asr elapsed: {elapsed_time * 1000:.2f} milliseconds")
    return result


"""
方法 - end - ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
"""

# 实例化FastAPI
app = FastAPI()

# 配置跨域中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """统一异常处理，返回标准化错误响应。

    Args:
        request (Request): 请求对象。
        exc (Exception): 捕获的异常对象。

    Returns:
        JSONResponse: 标准化的错误响应。
    """
    logger.error("Exception occurred", exc_info=True)
    if isinstance(exc, HTTPException):
        status_code = exc.status_code
        message = exc.detail
        data = ""
    elif isinstance(exc, RequestValidationError):
        status_code = HTTP_422_UNPROCESSABLE_ENTITY
        message = "Validation error: " + str(exc.errors())
        data = ""
    else:
        status_code = 500
        message = "Internal server error: " + str(exc)
        data = ""

    return JSONResponse(
        status_code=status_code,
        content=TranscriptionResponse(
            code=status_code,
            msg=message,
            data=data
        ).model_dump()
    )


@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    """
    处理WebSocket连接的音频流处理端点。
    """
    try:
        # 初始化配置和缓存
        chunk_size = int(config.chunk_size_ms * config.sample_rate / 1000)
        audio_buffer = np.array([], dtype=np.float32)
        audio_vad = np.array([], dtype=np.float32)
        # 初始化缓存和变量
        cache = {}
        cache_asr = {}
        last_vad_beg = last_vad_end = -1
        offset = 0
        buffer = b""
        # 解析查询参数
        query_params = parse_qs(websocket.scope['query_string'].decode())
        lang = query_params.get('lang', ['auto'])[0].lower()
        session = query_params.get('session')[0]  # 获取session id
        # 如果没有session id，就不执行后面的代码
        if session is None:
            logger.info("无 session id")
            return
        logger.info(f"session: {session}")
        # 如果会话列表的session为空，则创建一个空的对象
        if session not in session_list or session_list[session] is None:
            session_list[session] = {}
        #  接受WebSocket连接
        await websocket.accept()

        # 死循环获取音频数据
        while True:
            logger.info("websocket_endpoint-----------------")
            # 接收音频数据
            data = await websocket.receive_bytes()
            # 缓冲区数据处理
            buffer += data

            # 数据处理前的缓冲区校验
            if len(buffer) < 2:
                # 跳过处理
                continue
            # 转换字节数据为归一化的浮点数组
            audio_buffer = np.append(
                audio_buffer,
                np.frombuffer(buffer[:len(buffer) - (len(buffer) % 2)], dtype=np.int16).astype(np.float32) / 32767.0
            )

            # 保留未处理的奇数字节
            buffer = buffer[len(buffer) - (len(buffer) % 2):]

            # 分块处理音频数据
            while len(audio_buffer) >= chunk_size:
                # 获取音频块
                chunk = audio_buffer[:chunk_size]
                # 更新音频缓冲区
                audio_buffer = audio_buffer[chunk_size:]
                # 更新语音缓冲区
                audio_vad = np.append(audio_vad, chunk)
                logger.info(f"[websocket_endpoint] audio_len: {len(audio_vad)}; chunk_size: {chunk_size}")

                # 执行语音活动检测
                res = model_vad.generate(input=chunk, cache=cache, is_final=False, chunk_size=config.chunk_size_ms)
                if len(res[0]["value"]):
                    vad_segments = res[0]["value"]
                    for segment in vad_segments:
                        # 检测到语音开始
                        if segment[0] > -1:
                            last_vad_beg = segment[0]
                        # 检测到语音结束
                        if segment[1] > -1:
                            last_vad_end = segment[1]
                        # 处理语音活动片段
                        if last_vad_beg > -1 and last_vad_end > -1:
                            # 计算偏移量并更新last_vad_beg和last_vad_end
                            last_vad_beg -= offset
                            last_vad_end -= offset
                            offset += last_vad_end
                            beg = int(last_vad_beg * config.sample_rate / 1000)
                            end = int(last_vad_end * config.sample_rate / 1000)
                            logger.info(f"[vad segment] audio_len: {end - beg}")
                            # 调用asr函数进行语音识别
                            audio_vad1 = audio_vad[beg:end]
                            result = asr(audio_vad1, lang.strip(), cache_asr, True)
                            logger.info(f"asr response: {result}")
                            # 更新audio_vad和last_vad_beg和last_vad_end
                            audio_vad = audio_vad[end:]
                            # 重置语音缓冲区
                            last_vad_beg = last_vad_end = -1
                            if result is not None:
                                # 存储音频数据
                                spk_id = None
                                hit = False
                                if audio_vad1 is not None and len(audio_vad1) > 0:
                                    # 判断session_list[session]是否是空对象,如果是空对象就存入
                                    if isinstance(session_list[session], dict) and session_list[session] == {}:
                                        logger.info("直接存入音频波形数据")
                                        spk_id = str(uuid.uuid4())
                                        session_list[session] = {}
                                        session_list[session][spk_id] = {'name': '说话人1', 'data': audio_vad1,
                                                                         'sr': config.sample_rate}
                                    else:
                                        logger.info("通过speaker_verify 判断是否是同一个人")
                                        for key, value in list(session_list[session].items()):
                                            if speaker_verify(audio_vad1, value['data'], config.sv_thr):
                                                hit = True
                                                logger.info(f"[speaker_verify] hit: {hit}; {key}: {value}")
                                                spk_id = key
                                                # 如果命中就跳出循环
                                                break
                                            else:
                                                hit = False
                                                logger.info(f"[speaker_verify] hit: {hit}; {key}: {value}")
                                        if not hit:
                                            spk_id = str(uuid.uuid4())
                                            session_list[session][spk_id] = {
                                                'name': f'说话人{len(session_list[session]) + 1}',
                                                'data': audio_vad1,
                                                'sr': config.sample_rate
                                            }
                                print(session_list[session])
                                spk_set = {}
                                for key, value in session_list[session].items():
                                    spk_set[str(key)] = value['name']
                                logger.info(
                                    f"spk_id: {spk_id}\n"
                                    f"spk_set: {spk_set}\n"
                                    f"data: {result[0]['text']}"
                                )
                                # 发送最终识别结果
                                response = TranscriptionResponse(
                                    code=0,
                                    spk_id=spk_id if spk_id is not None else '无',
                                    spk_set=json.dumps(spk_set, ensure_ascii=False),
                                    # spk_id='',
                                    # spk_set='',
                                    info=json.dumps(result[0], ensure_ascii=False),
                                    data=format_str_v3(result[0]['text'])
                                    # data=format_str_v3(content)
                                )
                                await websocket.send_json(response.model_dump())

    except WebSocketDisconnect:
        logger.error("WebSocket 已断开连接")
    except Exception as e:
        logger.error(f"意外错误: {e}\n调用堆栈:\n{traceback.format_exc()}")
        await websocket.close()
    finally:
        # 资源清理
        audio_buffer = np.array([], dtype=np.float32)
        audio_vad1 = np.array([], dtype=np.float32)
        audio_vad = np.array([], dtype=np.float32)
        cache.clear()
        logger.info("在 WebSocket 断开连接后清理资源")


# 主启动
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI app with a specified port.")
    parser.add_argument('--port', type=int, default=27000, help='Port number to run the FastAPI app on.')
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
