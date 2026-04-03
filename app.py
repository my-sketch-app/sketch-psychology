import streamlit as st
import tempfile
import os
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
    import urllib.request
    model_path = "best.pt"
    
    if not os.path.exists(model_path):
        st.info("正在下载模型...")
        # 替换成你的网盘直链
        url = "https://ttttt.link/f/69cf9a42d7f53/best.pt"
        urllib.request.urlretrieve(url, model_path)
        st.success("模型下载完成")
    
    return YOLO(model_path)

# 物体到心理维度的映射
def map_to_psychology(detections):
    """将检测到的物体映射到心理维度"""
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

# AI心理分析
def ai_analysis(detections, scores, api_key):
    """调用硅基流动API进行心理分析"""
    if not api_key:
        return "请配置 API Key"
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.siliconflow.cn/v1"
    )
    
    # 构建提示词
    prompt = f"""你是一位专业的心理分析师，擅长房树人绘画心理分析。

用户画了一幅简笔画，检测到以下物体：
{detections}

心理维度评分：
{scores}

请给出专业的心理分析报告，包括：
1. 整体心理状态评估
2. 各维度详细解读
3. 给用户的具体建议
4. 关注等级（正常/需要关注/建议咨询）

请用温暖、专业的语气回答。"""

    try:
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[
                {"role": "system", "content": "你是一位温暖专业的心理分析师，擅长绘画心理分析。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI分析失败: {e}"

# 侧边栏说明
with st.sidebar:
    st.header("📖 使用说明")
    st.markdown("""
    1. 上传你的简笔画图片
    2. 点击「开始分析」按钮
    3. 系统会识别画中的物体
    4. AI会生成心理分析报告
    
    **建议画的元素：**
    - 人物（自我认知）
    - 房子（家庭安全感）
    - 树木（成长动力）
    - 太阳（积极情绪）
    """)
    st.markdown("---")
    st.caption("基于 YOLOv8 + 硅基流动大模型")

# 主界面
uploaded_file = st.file_uploader(
    "📤 上传简笔画",
    type=["jpg", "png", "jpeg"],
    help="支持 JPG、PNG 格式"
)

if uploaded_file:
    # 显示上传的图片
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(uploaded_file, caption="你的简笔画", use_container_width=True)
    
    with col2:
        if st.button("🎯 开始分析", type="primary"):
            with st.spinner("分析中，请稍候..."):
                # 保存临时文件
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                
                # 加载模型
                model = load_model()
                
                if model:
                    # YOLO检测（降低阈值）
                    results = model(tmp_path, conf=0.1)
                    
                    # 提取检测结果
                    detections = []
                    for r in results:
                        if r.boxes is not None:
                            for box in r.boxes:
                                conf = float(box.conf[0])
                                cls = int(box.cls[0])
                                name = model.names[cls]
                                detections.append({
                                    'object': name,
                                    'confidence': round(conf, 2)
                                })
                    
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
                        
                        # 显示分数
                        st.subheader("📊 心理维度评分")
                        for dim, score in scores.items():
                            bar = "█" * int(score * 20)
                            st.write(f"{dim}: {bar} {score:.2f}")
                        
                        # AI深度分析
                        st.subheader("🤖 AI心理分析报告")
                        api_key = st.secrets.get("SILICONFLOW_API_KEY", "")
                        analysis = ai_analysis(mapped, scores, api_key)
                        st.markdown(analysis)
                        
                    else:
                        st.warning("⚠️ 未检测到物体")
                        st.info("建议：\n1. 画得再清晰一些\n2. 尝试画人物、房子、树、太阳等元素\n3. 线条不要太细")
                else:
                    st.error("模型加载失败")
                
                # 清理临时文件
                os.unlink(tmp_path)

# 页脚
st.markdown("---")
st.caption("🎨 简笔画心理分析系统 | 技术：YOLOv8 + 硅基流动大模型")
