from js import document, navigator
from pyodide.ffi.wrappers import add_event_listener
import asyncio
import numpy as np
import posenet  # 假設我們有一個名為 posenet 的模組

# 設置全局變數
ctx = None
canvas_width = 0
canvas_height = 0
frame_count = 0

KEY_LEFT = 37
KEY_UP = 38
KEY_RIGHT = 39
KEY_DOWN = 40
KEY_SPACE = 32

_pressedKeys = {
    KEY_LEFT: False,
    KEY_UP: False,
    KEY_RIGHT: False,
    KEY_DOWN: False,
    KEY_SPACE: False,
}

# 設置視頻元素
video_element = document.createElement("video")
video_element.style.display = "none"  # 隱藏視頻元素
document.body.appendChild(video_element)


# 初始化攝像頭
async def setup_camera():
    stream = await navigator.mediaDevices.getUserMedia({"video": True})
    video_element.srcObject = stream
    await video_element.play()


# 捕捉當前幀
def capture_frame():
    ctx.drawImage(video_element, 0, 0, canvas_width, canvas_height)


# 初始化畫布和攝像頭
def init(width: int, height: int, canvas, scale: int = 1):
    global ctx, canvas_width, canvas_height
    ctx = canvas.getContext("2d", {"alpha": False})

    ctx.mozImageSmoothingEnabled = False
    ctx.webkitImageSmoothingEnabled = False
    ctx.msImageSmoothingEnabled = False
    ctx.imageSmoothingEnabled = False

    canvas.style.width = f"{width * scale}px"
    canvas.style.height = f"{height * scale}px"
    canvas.width = width * scale
    canvas.height = height * scale

    canvas_width = width * scale
    canvas_height = height * scale

    asyncio.ensure_future(setup_camera())  # 開始攝像頭

    ctx.clearRect(0, 0, width * scale, height * scale)

    add_event_listener(document, "keydown", _handle_input)
    add_event_listener(document, "keyup", _handle_input)


# 處理鍵盤事件
def _handle_input(e):
    global _pressedKeys
    if e.type == "keydown":
        _pressedKeys[e.keyCode] = True
    elif e.type == "keyup":
        _pressedKeys[e.keyCode] = False


# 加載 PoseNet 模型
posenet_model = None


async def load_posenet_model():
    global posenet_model
    if not posenet_model:
        posenet_model = await posenet.load()
    return posenet_model


# 檢測姿勢
async def detect_pose():
    global posenet_model
    if not posenet_model:
        posenet_model = await load_posenet_model()

    # 捕捉當前幀
    capture_frame()

    # 轉換幀為數據
    input_image = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    ctx.getImageData(0, 0, canvas_width, canvas_height).data = (
        input_image.flatten().tolist()
    )

    # 估計姿勢
    pose = await posenet_model.estimate_single_pose(
        input_image, {"flipHorizontal": False}
    )

    # 分析姿勢並設置鍵盤按鍵狀態
    for keypoint in pose["keypoints"]:
        if keypoint["score"] < 0.5:
            continue
        position = keypoint["position"]

        # 將關鍵點映射到按鍵
        if keypoint["part"] == "nose" and position["y"] < canvas_height * 0.3:
            _pressedKeys[KEY_UP] = True
        else:
            _pressedKeys[KEY_UP] = False

        if keypoint["part"] == "nose" and position["y"] > canvas_height * 0.7:
            _pressedKeys[KEY_DOWN] = True
        else:
            _pressedKeys[KEY_DOWN] = False

        if keypoint["part"] == "leftWrist" and position["x"] < canvas_width * 0.3:
            _pressedKeys[KEY_LEFT] = True
        else:
            _pressedKeys[KEY_LEFT] = False

        if keypoint["part"] == "rightWrist" and position["x"] > canvas_width * 0.7:
            _pressedKeys[KEY_RIGHT] = True
        else:
            _pressedKeys[KEY_RIGHT] = False

        if (
            keypoint["part"] in ["leftWrist", "rightWrist"]
            and position["y"] < canvas_height * 0.3
        ):
            _pressedKeys[KEY_SPACE] = True
        else:
            _pressedKeys[KEY_SPACE] = False


# 更新函數，用於每一幀調用
def update():
    global frame_count
    frame_count += 1
    asyncio.ensure_future(detect_pose())  # 更新姿勢檢測


# 設置初始化
init(640, 480, document.querySelector("#myCanvas"), 1)
