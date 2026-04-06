import streamlit as st
import tempfile
import os
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
    # 优先使用训练好的模型，如果没有则使用官方预训练模型
    if os.path.exists("best.pt"):
        return YOLO("best.pt")
    else:
        return YOLO("yolov8n.pt")

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
        obj = d['object']
        if obj in mapping:
            result.append({
                'object': obj,
                'confidence': d['confidence'],
                'dimension': mapping[obj]
            })
    return result

# 侧边栏说明
with st.sidebar:
    st.header("📖 使用说明")
    st.markdown("""
    1. 上传你的简笔画图片
    2. 点击「开始分析」按钮
    3. 系统会识别画中的物体
    4. 查看心理分析结果
    
    **建议画的元素：**
    - 人物（自我认知）
    - 房子（家庭安全感）
    - 树木（成长动力）
    - 太阳（积极情绪）
    """)

# 主界面
uploaded_file = st.file_uploader("上传简笔画", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="你的简笔画", width=300)
    
    if st.button("开始分析"):
        with st.spinner("分析中，请稍候..."):
            # 保存临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            
            # 加载模型并检测
            model = load_model()
            results = model(tmp_path, conf=0.3)
            
            # 提取检测结果
            detections = []
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        name = model.names[cls]
                        detections.append({'object': name, 'confidence': round(conf, 2)})
            
            # 显示检测结果
            if detections:
                st.success(f"✅ 检测到 {len(detections)} 个物体")
                
                # 心理映射
                mapped = map_to_psychology(detections)
                
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
                
                # 简单分析建议
                st.subheader("📋 简要分析")
                suggestions = []
                if scores.get('自我认知', 0) < 0.4:
                    suggestions.append("自我表达较少，可以多鼓励表达自己的想法")
                if scores.get('家庭安全感', 0) < 0.3:
                    suggestions.append("家庭相关元素较少")
                if scores.get('成长动力', 0) > 0.7:
                    suggestions.append("成长动力充足，积极向上")
                if scores.get('积极情绪', 0) > 0.7:
                    suggestions.append("情绪状态积极乐观")
                if scores.get('压力感', 0) > 0.6:
                    suggestions.append("压力相关元素较多，建议适当放松")
                
                if suggestions:
                    for s in suggestions:
                        st.write(f"• {s}")
                else:
                    st.write("心理状态整体良好")
            else:
                st.warning("未检测到物体")
                st.info("建议：画得再清晰一些，尝试画人物、房子、树、太阳等元素")
            
            # 清理临时文件
            os.unlink(tmp_path)

st.markdown("---")
st.caption("🎨 简笔画心理分析系统 | 基于 YOLOv8")
