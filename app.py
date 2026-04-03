# app.py
import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
from datetime import datetime
from openai import OpenAI

# 页面配置
st.set_page_config(
    page_title="简笔画心理分析系统",
    page_icon="🎨",
    layout="wide"
)

# 标题
st.title("🎨 简笔画心理分析系统")
st.markdown("---")

# 侧边栏说明
with st.sidebar:
    st.header("📖 使用说明")
    st.markdown("""
    1. 上传你的简笔画图片
    2. 点击"开始分析"按钮
    3. AI会分析你的心理状态
    4. 查看详细报告
    
    **提示：** 可以画人、房子、树、太阳等元素
    """)
    st.markdown("---")
    st.caption("基于 YOLO + 硅基流动大模型")

# 加载模型
@st.cache_resource
def load_model():
    """加载YOLO模型（缓存，只加载一次）"""
    model_path = "runs/models/sketch_model/weights/best.pt"
    if not os.path.exists(model_path):
        st.error(f"模型文件不存在: {model_path}")
        return None
    return YOLO(model_path)

# 初始化API客户端
@st.cache_resource
def get_api_client():
    """获取API客户端"""
    # 方式1：从secrets读取（部署时用）
    # 方式2：从环境变量读取（本地测试用）
    api_key = os.environ.get("SILICONFLOW_API_KEY", "")
    if not api_key:
        # 本地测试时，可以手动输入（注意：不要提交到GitHub！）
        api_key = st.secrets.get("SILICONFLOW_API_KEY", "")
    
    if not api_key:
        st.warning("⚠️ 请配置API Key")
        return None
    
    return OpenAI(
        api_key=api_key,
        base_url="https://api.siliconflow.cn/v1"
    )

# 心理映射
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
    
    mapped = []
    for d in detections:
        obj = d['object']
        if obj in mapping:
            mapped.append({
                'object': obj,
                'confidence': d['confidence'],
                'dimension': mapping[obj]
            })
    return mapped

# 计算分数
def calculate_scores(mapped_detections):
    """计算心理维度分数"""
    scores = {
        '自我认知': 0.5,
        '家庭安全感': 0.5,
        '成长动力': 0.5,
        '积极情绪': 0.5,
        '压力感': 0.5,
        '情感表达': 0.5,
        '社交倾向': 0.5
    }
    
    for d in mapped_detections:
        dim = d['dimension']
        if dim in scores:
            scores[dim] = min(1.0, scores[dim] + d['confidence'] * 0.25)
    
    return scores

# 调用AI分析
def ai_analysis(detections, scores, client):
    """调用硅基流动大模型进行深度分析"""
    
    prompt = f"""你是一位专业的心理分析师。

用户画了一幅简笔画，检测到以下物体：
{detections}

心理维度评分：
{scores}

请给出：
1. 整体心理状态评估
2. 各维度解读
3. 具体建议
4. 关注等级（正常/需要关注/建议咨询）

请用温暖专业的语气回答。"""

    try:
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[
                {"role": "system", "content": "你是专业的心理分析师，回答温暖有建设性。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI分析失败: {e}"

# 主界面
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 上传简笔画")
    uploaded_file = st.file_uploader(
        "点击或拖拽上传图片",
        type=["jpg", "png", "jpeg"],
        help="支持JPG、PNG格式"
    )
    
    if uploaded_file:
        st.image(uploaded_file, caption="你画的简笔画", use_container_width=True)

with col2:
    st.subheader("🔍 分析结果")
    
    if uploaded_file and st.button("🎯 开始分析", type="primary"):
        # 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        # 显示进度
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 1. 加载模型
        status_text.text("正在加载AI模型...")
        progress_bar.progress(20)
        model = load_model()
        
        if model is None:
            st.error("模型加载失败")
        else:
            # 2. YOLO检测
            status_text.text("正在识别简笔画元素...")
            progress_bar.progress(40)
            results = model(tmp_path, conf=0.3)
            
            # 提取检测结果
            detections = []
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = model.names[cls]
                    detections.append({'object': name, 'confidence': round(conf, 2)})
            
            # 3. 心理映射
            status_text.text("正在分析心理特征...")
            progress_bar.progress(60)
            mapped = map_to_psychology(detections)
            scores = calculate_scores(mapped)
            
            # 显示检测结果
            if detections:
                st.success(f"✅ 检测到 {len(detections)} 个物体")
                for d in detections:
                    st.write(f"  • {d['object']}: {d['confidence']}")
            else:
                st.warning("⚠️ 未检测到明显物体，建议画更清晰的简笔画")
            
            # 4. 调用AI
            status_text.text("正在生成心理报告...")
            progress_bar.progress(80)
            
            client = get_api_client()
            if client:
                analysis = ai_analysis(mapped, scores, client)
                
                # 5. 显示结果
                progress_bar.progress(100)
                status_text.text("分析完成！")
                
                st.markdown("---")
                st.subheader("📋 心理分析报告")
                st.markdown(analysis)
                
                # 显示评分表格
                st.subheader("📊 心理维度评分")
                for dim, score in scores.items():
                    st.progress(score, text=f"{dim}: {score:.0%}")
            else:
                st.error("API连接失败，请检查配置")
        
        # 清理临时文件
        os.unlink(tmp_path)

# 页脚
st.markdown("---")
st.caption("🎨 简笔画心理分析系统 | 基于YOLO + 硅基流动大模型")