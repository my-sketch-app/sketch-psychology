import streamlit as st
import requests
import json
from PIL import Image
import io

st.set_page_config(page_title="简笔画心理分析系统", page_icon="🎨")
st.title("🎨 简笔画心理分析系统")
st.markdown("上传你的简笔画，系统会分析心理状态")

# 侧边栏说明
with st.sidebar:
    st.header("📖 使用说明")
    st.markdown("""
    1. 上传你的简笔画图片
    2. 点击「开始分析」按钮
    3. 系统会识别画中的物体
    4. 查看心理分析结果
    """)

# 物体到心理维度的映射
def map_to_psychology(detections):
    mapping = {
        'person': '自我认知',
        'house': '家庭安全感',
        'tree': '成长动力',
        'sun': '积极情绪',
        'cloud': '压力感',
        'flower': '情感表达',
        'animal': '社交倾向'
    }
    result = []
    for d in detections:
        obj = d.get('label', '')
        if obj in mapping:
            result.append({
                'object': obj,
                'confidence': d.get('confidence', 0),
                'dimension': mapping[obj]
            })
    return result

# 主界面
uploaded_file = st.file_uploader("上传简笔画", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="你的简笔画", width=300)
    
    if st.button("开始分析"):
        with st.spinner("分析中，请稍候..."):
            try:
                # 读取图片
                image_bytes = uploaded_file.read()
                
                # 使用 Hugging Face 的免费推理 API（物体检测）
                API_URL = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"
                headers = {"Authorization": "Bearer hf_xxxxxxxxxxxxxxxxxxxxx"}  # 需要注册获取 token
                
                # 由于需要 token，这里先用模拟数据演示
                # 实际使用时需要注册 Hugging Face 并获取免费 token
                
                # 模拟检测结果（实际使用时替换为 API 调用）
                detections = [
                    {'label': 'tree', 'confidence': 0.88},
                    {'label': 'tree', 'confidence': 0.81},
                    {'label': 'sun', 'confidence': 0.71}
                ]
                
                # 心理映射
                mapped = map_to_psychology(detections)
                
                if mapped:
                    st.success(f"✅ 检测到 {len(mapped)} 个物体")
                    
                    st.subheader("🔍 检测结果")
                    for m in mapped:
                        st.write(f"• **{m['object']}** ({m['confidence']:.2f}) → {m['dimension']}")
                    
                    # 计算维度分数
                    scores = {}
                    for m in mapped:
                        dim = m['dimension']
                        if dim not in scores:
                            scores[dim] = 0
                        scores[dim] = max(scores[dim], m['confidence'])
                    
                    st.subheader("📊 心理维度评分")
                    for dim, score in scores.items():
                        bar = "█" * int(score * 20)
                        st.write(f"{dim}: {bar} {score:.2f}")
                    
                    st.subheader("📋 简要分析")
                    suggestions = []
                    if scores.get('自我认知', 0) < 0.4:
                        suggestions.append("自我表达较少，可以多鼓励表达自己的想法")
                    if scores.get('成长动力', 0) > 0.7:
                        suggestions.append("成长动力充足，积极向上")
                    if scores.get('积极情绪', 0) > 0.7:
                        suggestions.append("情绪状态积极乐观")
                    
                    if suggestions:
                        for s in suggestions:
                            st.write(f"• {s}")
                    else:
                        st.write("心理状态整体良好")
                else:
                    st.warning("未检测到明显物体")
                    
            except Exception as e:
                st.error(f"分析失败: {e}")

st.markdown("---")
st.caption("🎨 简笔画心理分析系统")
