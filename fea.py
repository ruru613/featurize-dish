import streamlit as st
import sys
import pathlib
import asyncio
from PIL import Image
from fastai.vision.all import *
from streamlit.scriptrunner import add_script_run_ctx

# 确保Python版本兼容
if sys.version_info >= (3, 13):
    st.error("⚠️ 当前 Python 版本为 3.13+，可能与 fastai 不兼容。建议使用 Python 3.11。")
    st.stop()

# 更健壮的事件循环处理
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# 将事件循环与Streamlit线程绑定
add_script_run_ctx(loop)

@st.cache_resource
def load_model():
    try:
        model_path = pathlib.Path(__file__).parent / "dish.pkl"
        return load_learner(model_path)
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None

# 加载模型
model = load_model()

st.title("Doraemon 与 Walle 分类器")
st.write("上传一张图片，看看它是 Doraemon 还是 Walle！")

# 图片上传和处理
uploaded_file = st.file_uploader("选择一张图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # 使用PIL加载图片
        image = PILImage.create(uploaded_file)
        st.image(image, caption="上传的图片", use_column_width=True)
        
        # 检查模型是否成功加载
        if model is not None:
            pred, pred_idx, probs = model.predict(image)
            st.success(f"预测结果：{pred}")
            st.write(f"概率：{probs[pred_idx]:.04f}")
        else:
            st.warning("模型未成功加载，无法进行预测。")
    except Exception as e:
        st.error(f"预测过程中发生错误: {e}")