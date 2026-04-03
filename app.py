# app.py - 简笔画心理分析系统（部署版）
import streamlit as st
import tempfile
import os
from datetime import datetime
from openai import OpenAI
from ultralytics import YOLO

# 页面配置
st.set_page_config(
    page_title="简笔画心理分析系统",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 简笔画心理分析系统")
st.markdown("上传你的简笔画，AI会分析你的心理状态")

# 加载模型
@st.cache_resource
def load_model():
    import os
    
    # 检查模型文件大小
    model_path = "best.pt"
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)
        st.write(f"模型文件大小: {file_size / 1024 / 1024:.2f} MB")
    else:
        st.error("模型文件不存在")
        return None
    
    return YOLO(model_path)

# 心理映射
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
        obj = d['object']
        if obj in mapping:
            result.append({
                'object': obj,
                'confidence': d['confidence'],
                'dimension': mapping[obj]
            })
    return result

# AI分析
def ai_analysis(detections, scores, api_key):
    client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")
    
    prompt = f"""你是一位心理分析师。用户画了简笔画，检测到：{detections}，评分：{scores}。
请给出：1.整体评估 2.维度解读 3.建议 4.关注等级"""
    
    try:
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"分析失败: {e}"

# 界面
uploaded_file = st.file_uploader("上传简笔画", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="你的简笔画", width=300)
    
    if st.button("开始分析"):
        with st.spinner("分析中..."):
            # 保存临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            
            # YOLO检测
            model = load_model()
            if model:
            # 检测
results = model(tmp_path, conf=0.1)  # 降低阈值
detections = []
for r in results:
    if r.boxes is not None:
        st.write(f"原始检测: {len(r.boxes)} 个物体")
        for box in r.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            name = model.names[cls]
            st.write(f"  {name}: {conf:.2f}")
            
            if conf > 0.1:  # 显示所有
                detections.append({
                    'object': name,
                    'confidence': conf
                })
    else:
        st.write("原始检测: 0 个物体")

st.write(f"最终检测: {len(detections)} 个物体")
                
                # 心理映射
                mapped = map_to_psychology(detections)
                scores = {d['dimension']: d['confidence'] for d in mapped}
                
                # AI分析
                api_key = st.secrets.get("SILICONFLOW_API_KEY", "")
                if api_key:
                    analysis = ai_analysis(mapped, scores, api_key)
                    st.subheader("📋 心理分析报告")
                    st.write(analysis)
                else:
                    st.warning("请配置API Key")
            
            os.unlink(tmp_path)

st.markdown("---")
st.caption("基于 YOLO + 硅基流动大模型")
