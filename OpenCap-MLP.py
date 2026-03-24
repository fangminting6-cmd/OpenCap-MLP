import streamlit as st
import os
import requests
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import io
import zipfile
import traceback
import streamlit as st

# ===================== 0. 全局配置 =====================
DEFAULT_MODEL_NAME = "final_MLP_multioutput_model.pkl"  # <--- 新增：定义仓库里的模型文件名
# ===================== 0. 全局配置 (新加) =====================
ADVICE_MAP = {
    "HFA": (
        " 🏋️‍♂️ 髋关节屈曲 (HFA) 优化方案\n\n"
        "**⚠️ 风险机制：** 髋屈不足导致下肢刚性着地，冲击力无法通过大肌群有效吸收。\n\n"
        "**💡 训练建议：**\n"
        "* **基础 (髋铰链)：** 罗马尼亚硬拉 (RDL)。强调臀部向后推，膝盖微屈但不前推，保持背部平直。*(3组 x 8-12次)*\n"
        "* **进阶 (爆发力)：** 壶铃摆动 (Kettlebell Swing)。利用臀部爆发式伸展“弹”起壶铃，在动作顶点收紧臀肌。*(3组 x 15次)*\n"
        "* **动态 (落地控制)：** 高台深落地 (Drop Landing)。从 30cm 高台迈下，触地瞬间迅速“向后坐”，像坐椅子一样。*(3组 x 5-8次)*\n\n"
        "**🎯 技术要点：** 触地瞬间增加“向后坐”的缓冲动作，延长力传导时间。"
    ),
    "HAA": (
        " 🛡️ 髋关节内收 (HAA) 优化方案\n\n"
        "**⚠️ 风险机制：** 髋内收过大会诱发膝外翻（动态内扣），是 ACL (前交叉韧带) 损伤的核心风险因素。\n\n"
        "**💡 训练建议：**\n"
        "* **激活 (臀中肌)：** 弹力带蚌式开合。保持骨盆绝对稳定不向后翻转，体会臀部侧上方的酸胀感。*(3组 x 15-20次/侧)*\n"
        "* **力量 (抗内扣)：** 怪兽步 (Monster Walk)。保持 1/4 蹲姿，侧向行走时主动对抗弹力带拉力，绝不允许膝盖内扣。*(3组 x 15米/方向)*\n"
        "* **功能性 (单腿稳定)：** 单腿下蹲。对着镜子练习，全程盯住膝盖，确保髌骨中心对准第2-3足趾。*(3组 x 8-10次/侧)*\n\n"
        "**🎯 技术要点：** 确保跳跃/落地过程中，膝盖中心始终对准脚尖方向。"
    ),
    "KFA": (
        " 🦵 膝关节屈曲 (KFA) 优化方案\n\n"
        "**⚠️ 风险机制：** 着地时膝关节过直（伸膝位）会产生巨大的前向剪切力，容易导致关节损伤。\n\n"
        "**💡 训练建议：**\n"
        "* **基础 (等长收缩)：** 靠墙静蹲。重心压在足中和脚后跟，下背部贴紧墙壁，感受大腿前侧发力。*(3-4组 x 45-60秒)*\n"
        "* **离心 (制动力)：** 诺迪克后屈腿 (Nordic Curls)。固定双脚，身体挺直缓慢前倾，用大腿后侧控制下降速度。*(3组 x 5-8次)*\n"
        "* **专项 (吸震反应)：** 连续跳深缓冲。从高处跳下，触地瞬间主动屈膝屈髋“吸收”冲击，落地声越轻越好。*(3组 x 6-8次)*\n\n"
        "**🎯 技术要点：** 触地时主动通过屈膝动作来“吸震”，做到落地无声（像猫一样）。"
    ),
    "ADF": (
        " 🦶 踝关节背屈 (ADF) 优化方案\n\n"
        "**⚠️ 风险机制：** 踝关节活动度受限会限制整体下肢缓冲行程，迫使膝关节进行代偿。\n\n"
        "**💡 训练建议：**\n"
        "* **松解 (软组织)：** 泡沫轴深度松解。重点滚压小腿肚和靠近跟腱的区域，痛点停留 20 秒并转动脚踝。*(双侧各 1-2 分钟)*\n"
        "* **拉伸 (活动度)：** 靠墙弓箭步拉伸 (Knee-to-Wall)。足跟死死踩住地面，膝盖尽量向前顶触碰墙壁。*(3组 x 每侧 30秒)*\n"
        "* **动态 (受压灵活性)：** 负重高脚杯深蹲停留。在最深蹲姿下，将重物压在单侧大腿前侧，迫使膝盖超过脚尖。*(3组 x 每侧压迫 20秒)*\n\n"
        "**🎯 技术要点：** 提高踝关节灵活性，确保足跟在落地支撑期能平稳过渡。"
    ),
    "FPA": (
        " 🧭 足偏角 (FPA) 优化方案\n\n"
        "**⚠️ 风险机制：** 过度内八或外八会改变下肢动力链力线，增加膝关节的扭转应力。\n\n"
        "**💡 训练建议：**\n"
        "* **平衡 (足底感知)：** 赤足单腿闭眼站立。感受前脚掌内外侧和足跟构成的“足底三角”均匀受力。*(3组 x 30-45秒/侧)*\n"
        "* **整合 (力线纠正)：** 弓箭步走 (Lunge Walk)。强制盯住前后脚，必须 100% 朝向正前方，拒绝脚尖外撇。*(3组 x 每侧 10-12步)*\n"
        "* **敏捷 (动态对齐)：** 十字象限跳 (Dot Drill)。在四个象限间快速跳跃，关注落地时双脚的平行度，避免疲劳时外八。*(3组 x 20-30秒)*\n\n"
        "**🎯 技术要点：** 保持足部纵轴与行进方向基本一致，消除多余的旋转负荷。"
    ),
    "TFA": (
        " ⚖️ 躯干倾斜 (TFA) 优化方案\n\n"
        "**⚠️ 风险机制：** 躯干不稳会导致重心偏移，产生巨大的膝关节冠状面力矩，破坏身体平衡。\n\n"
        "**💡 训练建议：**\n"
        "* **静态 (侧向抗力)：** 侧桥 (Side Plank)。从头到脚跟呈一条直线，主动将骨盆顶起，核心绷紧不塌腰。*(3组 x 30-45秒/侧)*\n"
        "* **动态 (抗侧屈)：** 单臂农夫行走 (Suitcase Carry)。单手提重物行走，刻意对抗重力，保持双肩水平不向一侧倾斜。*(3组 x 每侧 20米)*\n"
        "* **专项 (动态中立)：** 滑冰跳 (Skater Jumps)。左右单腿横向跳跃，落地停顿 2 秒，控制上半身不要过度侧倾。*(3组 x 每侧 8-10次)*\n\n"
        "**🎯 技术要点：** 维持脊柱中立位稳定，减少触地瞬间身体重心的左右摇晃。"
    )
}
# 各关节在触地瞬间(IC)的正常参考角度（度），需根据你的实验标准微调
NORMAL_VALUES = {
    "HFA": 28.0,  # 髋关节屈曲正常值
    "HAA": 5,   # 髋关节内收正常值
    "KFA": 26.0,  # 膝关节屈曲正常值
    "ADF": 10.0,  # 踝关节背屈正常值
    "FPA": 10.0,   # 足偏角正常值
    "TFA": 25.0   # 躯干侧倾正常值
}

# ===================== 1. 身份验证逻辑 (集成版) =====================
def get_opencap_token():
    """从 Secrets 获取凭据并向 OpenCap 请求 Token"""
    try:
        # 从 Streamlit Cloud Secrets 读取
        username = st.secrets["OPENCAP_USER"]
        password = st.secrets["OPENCAP_PASS"]
    except Exception:
        st.error("❌ 错误：未在 Streamlit Secrets 中找到 OPENCAP_USER 或 OPENCAP_PASS")
        st.stop()

    login_url = "https://api.opencap.ai/login/"
    try:
        resp = requests.post(login_url, data={'username': username, 'password': password})
        if resp.status_code == 200:
            return resp.json().get('token')
        else:
            st.error(f"❌ 登录 OpenCap 失败 (状态码: {resp.status_code}): {resp.text}")
            st.stop()
    except Exception as e:
        st.error(f"❌ 网络请求异常: {e}")
        st.stop()

# ===================== 2. Streamlit 页面配置 =====================
st.set_page_config(page_title="ACL 损伤风险预测", layout="wide")

st.title("🏃‍♂️ ACL损伤风险快速筛查系统")

# 侧边栏配置
st.sidebar.header("⚙️ 配置参数")
session_id = st.sidebar.text_input("Session ID", value="995d44f9-022a-449b-9dd2-2424318c3f54")
trial_keyword = st.sidebar.text_input("动作试次名称", value="single-jumpGR_6_1")
# 修改点：将上传组件设为可选，不再是必填项
model_file = st.sidebar.file_uploader("上传自定义模型 (不上传则使用内置默认模型)", type=["pkl"])

# ===================== 3. 核心分析逻辑 =====================
def run_analysis(sid, keyword, model_obj):
    try:
        with st.status("🔍 正在执行分析流程...", expanded=True) as status:
            st.write("正在获取身份令牌...")
            token = get_opencap_token()
            
            st.write("正在从 OpenCap 下载 Session 数据...")
            headers = {"Authorization": f"Token {token}"}
            url = f"https://api.opencap.ai/sessions/{sid}/download/"
            
            resp = requests.get(url, headers=headers)
            if resp.status_code != 200:
                st.error(f"❌ 下载失败，状态码: {resp.status_code}")
                return

            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                all_files = z.namelist()
                # 筛选动态试次的 MOT 文件
                all_mots = [f for f in all_files if f.endswith('.mot') and 'Kinematics' in f and 'static' not in f.lower()]
                
                if not all_mots:
                    st.error("❌ 该 Session 中未发现有效的 MOT 运动数据文件。")
                    return

                # 根据关键词选择试次
                selected_mot = next((m for m in all_mots if keyword.lower() in m.lower()), all_mots[0])
                trial_id = os.path.basename(selected_mot).replace('.mot', '')
                
                # 寻找对应的 TRC 文件
                try:
                    selected_trc = [f for f in all_files if f.endswith('.trc') and trial_id in f and 'MarkerData' in f][0]
                except IndexError:
                    st.error(f"❌ 未找到与试次 {trial_id} 对应的 TRC 标记点数据。")
                    return

                st.write(f"正在解析试次数据: {trial_id}...")
                with z.open(selected_trc) as f:
                    df_trc = pd.read_csv(f, sep='\t', skiprows=6, header=None)
                with z.open(selected_mot) as f:
                    content = f.read().decode('utf-8').splitlines()
                    header_idx = next(i for i, line in enumerate(content) if 'time' in line.lower() and '\t' in line)
                    df_mot = pd.read_csv(io.StringIO('\n'.join(content[header_idx:])), sep='\t')

            # --- IC 瞬间定位 (最高点后首个回弹点) ---
            # 索引 54 对应 RBigToe 的垂直轴
            y_values = df_trc.iloc[:, 54].values 
            apex_idx = np.argmax(y_values)
            ic_idx = apex_idx
            for i in range(apex_idx + 1, len(y_values)):
                if y_values[i] < 0.15 and y_values[i] > y_values[i-1]:
                    ic_idx = i
                    break
            
            # --- 特征提取 ---
            row = df_mot.iloc[ic_idx].copy()
            feature_names = ["HFA", "HAA", "KFA", "ADF", "FPA", "TFA"]
            feature_values = [
                row.get('hip_flexion_r', 0),
                row.get('hip_adduction_r', 0),
                row.get('knee_angle_r', 0),
                row.get('ankle_angle_r', 0),
                row.get('subtalar_angle_r', 0) * -1,
                row.get('lumbar_extension', 0) * -1
            ]
            features_array = np.array([feature_values])
            
            loaded_package = joblib.load(model_obj)
            model = loaded_package['model']
            scaler_X = loaded_package['scaler_X']
            scaler_y = loaded_package['scaler_y']
            background_data = loaded_package['background_data']

            # 💡 修复 2：对输入特征进行标准化 (必须与训练时一致)
            input_df_raw = pd.DataFrame(features_array, columns=feature_names)
            input_scaled = scaler_X.transform(input_df_raw)
            
            # 使用标准化后的数据预测，并使用 scaler_y 逆转换回真实物理量纲
            pred_scaled = model.predict(input_scaled)
            pred_real = scaler_y.inverse_transform(pred_scaled)
            
            # 获取双输出的得分
            score_acl = float(pred_real[0][0])
            score_kneeload = float(pred_real[0][1])
            
            status.update(label="✅ 分析完成！", state="complete")
        # --- 结果展示面板 ---
        st.divider()
        
        # 💡 1. 设定两个指标的高风险阈值 (请根据你的实验标准修改下面的数字)
        ACL_THRESHOLD = 2.45
        KNEE_LOAD_THRESHOLD = 3.0  # <--- ⚠️ 请把 3.0 改为你判定 knee-load 高风险的实际标准
        
        # 💡 2. 分别判断两个指标是否超标
        is_acl_risk = score_acl >= ACL_THRESHOLD
        is_knee_risk = score_kneeload >= KNEE_LOAD_THRESHOLD
        
        # 💡 3. 总体风险判定：只要有一个指标超标，总体动作就判定为“高风险”
        is_overall_risk = is_acl_risk or is_knee_risk
        overall_text = "🚨 总体高风险" if is_overall_risk else "✅ 总体低风险"
        overall_color = "#d63031" if is_overall_risk else "#27ae60"

        # 调整为 3 列，展示指标的同时加上超标警告小标签
        m_col1, m_col2, m_col3 = st.columns(3)
        
        with m_col1:
            # 如果 ACL 超标，显示红色的 [超标] 警告
            acl_warning = "<span style='color:#d63031; font-size:18px; font-weight:bold;'> (超标)</span>" if is_acl_risk else ""
            st.markdown(f"### ACL 应力: <span style='color:#2d3436;'>{score_acl:.2f}</span>{acl_warning}", unsafe_allow_html=True)
            
        with m_col2:
            # 如果 Knee-load 超标，显示红色的 [超标] 警告
            knee_warning = "<span style='color:#d63031; font-size:18px; font-weight:bold;'> (超标)</span>" if is_knee_risk else ""
            st.markdown(f"### Knee Load: <span style='color:#2d3436;'>{score_kneeload:.2f}</span>{knee_warning}", unsafe_allow_html=True)
            
        with m_col3:
            # 展示总体风险判定
            st.markdown(f"### 风险判定: <span style='color:{overall_color};'>{overall_text}</span>", unsafe_allow_html=True)
        # --- SHAP 可视化 ---
        st.subheader("📊 关键动作特征贡献分析 (SHAP)")
        
        input_scaled_df = pd.DataFrame(scaler_X.transform(input_df_raw), columns=scaler_X.feature_names_in_)
        input_sel = input_scaled_df[important_features]
        
        explainer = shap.KernelExplainer(model.predict, background_data)
        shap_values_raw = explainer.shap_values(input_sel)
        
        # 💡 终极兼容：处理不同版本 SHAP 返回的多输出数据结构差异
        if isinstance(shap_values_raw, list):
            # 传统 SHAP 行为：返回 List，每个目标一个二维数组 [ (1, features), (1, features) ]
            val_acl = shap_values_raw[0][0]
            val_kneeload = shap_values_raw[1][0]
        else:
            # 新版 SHAP 行为：返回 3D 数组 (samples, features, outputs)
            if len(shap_values_raw.shape) == 3:
                val_acl = shap_values_raw[0, :, 0]
                val_kneeload = shap_values_raw[0, :, 1]
            else:
                # 极端兜底情况
                val_acl = shap_values_raw[0]
                val_kneeload = shap_values_raw[0]

        # 兼容 expected_value 的结构差异
        if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1:
            expected_val_acl = explainer.expected_value[0]
            expected_val_kneeload = explainer.expected_value[1]
        else:
            expected_val_acl = explainer.expected_value
            expected_val_kneeload = explainer.expected_value
            
        # 提取真实物理角度用于图表展示
        input_raw_sel = input_df_raw[important_features].iloc[0].values
        
        # 组装 ACL 的 Explanation 对象
        exp_acl = shap.Explanation(
            values=val_acl,          
            base_values=expected_val_acl,           
            data=input_raw_sel,                     
            feature_names=important_features        
        )
        
        # 组装 Knee-load 的 Explanation 对象
        exp_kneeload = shap.Explanation(
            values=val_kneeload,          
            base_values=expected_val_kneeload,
            data=input_raw_sel,
            feature_names=important_features
        )

        tab_acl, tab_kneeload = st.tabs(["🦵 ACL SHAP 解释图", "🏋️‍♂️ knee-load SHAP 解释图"])
        def draw_shap_plots(exp_obj, exp_val, target_name):
            st.markdown(f"**{target_name} - 瀑布图 (Waterfall):** 展示各特征对基准值的累加贡献")
            fig_wf, ax_wf = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(exp_obj, max_display=10, show=False)
            plt.tight_layout()
            st.pyplot(fig_wf)

            st.markdown("---")
            
            st.markdown(f"**{target_name} - 力图 (Force Plot):** 展示特征之间相互“推拉”的过程")
            shap.force_plot(
                exp_val, 
                exp_obj.values, 
                input_df_raw.iloc[0,:], 
                matplotlib=True, 
                show=False,
                plot_cmap=["#ff0051", "#008bfb"]
            )
            st.pyplot(plt.gcf(), clear_figure=True)

        with tab_acl:
            draw_shap_plots(exp_acl, expected_val_acl, "ACL")

        with tab_kneeload:
            draw_shap_plots(exp_kneeload, expected_val_kneeload, "knee-load")
       # --- 动态建议生成逻辑 ---
      # 💡 修复 3：使用 exp_acl.values 生成基于 ACL 的改善建议
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'contribution': exp_acl.values,  # <--- 明确使用 ACL 的贡献值
            'actual_value': feature_values   
        })
        
        risk_factors = shap_df[shap_df['contribution'] > 0].sort_values(by='contribution', ascending=False)

        with st.expander("📋 针对性动作改善建议 (基于 ACL 风险)", expanded=True):
            st.markdown(f"**分析对象**: `{trial_id}`  |  **触地瞬间 (IC)**: 第 {ic_idx+1} 帧")
            
            if not risk_factors.empty:
                st.markdown("#### ⚠️ 高风险动作特征偏差分析 (实测值 vs 正常值)")
                
                fig_db, ax_db = plt.subplots(figsize=(3.0, len(risk_factors) * 0.25 + 0.8), dpi=200)
                y_labels = []
                y_ticks = []
                
                for idx, (index, row) in enumerate(risk_factors[::-1].iterrows()):
                    f_name = row['feature']
                    actual_val = row['actual_value']
                    normal_val = NORMAL_VALUES.get(f_name, 0)
                    y = idx
                    
                    ax_db.plot([actual_val, normal_val], [y, y], color='#dcdde1', zorder=1, lw=1)
                    ax_db.scatter(normal_val, y, color='#008bfb', s=15, zorder=2, label='Normal Value' if idx==len(risk_factors)-1 else "")
                    ax_db.scatter(actual_val, y, color='#ff0051', s=15, zorder=2, label='Actual Risk' if idx==len(risk_factors)-1 else "")
                    
                    left_val, right_val = min(actual_val, normal_val), max(actual_val, normal_val)
                    offset = max(abs(actual_val - normal_val) * 0.1, 0.5) 
                    
                    if actual_val < normal_val:
                        ax_db.text(actual_val - offset, y, f"{actual_val:.1f}", va='center', ha='right', fontsize=6, color='#ff0051', fontweight='bold')
                        ax_db.text(normal_val + offset, y, f"{normal_val:.1f}", va='center', ha='left', fontsize=6, color='#008bfb')
                    else:
                        ax_db.text(normal_val - offset, y, f"{normal_val:.1f}", va='center', ha='right', fontsize=6, color='#008bfb')
                        ax_db.text(actual_val + offset, y, f"{actual_val:.1f}", va='center', ha='left', fontsize=6, color='#ff0051', fontweight='bold')
                    
                    y_labels.append(f_name)
                    y_ticks.append(y)
                
                ax_db.set_yticks(y_ticks)
                ax_db.set_yticklabels(y_labels, fontsize=7, fontweight='bold', color='#2d3436')
                ax_db.set_xlabel("Angle (Degree)", fontsize=7, color='#636e72')
                ax_db.tick_params(axis='x', labelsize=6)
                
                ax_db.spines['top'].set_visible(False)
                ax_db.spines['right'].set_visible(False)
                ax_db.spines['left'].set_visible(False)
                ax_db.spines['bottom'].set_color('#b2bec3')
                ax_db.grid(axis='x', linestyle='--', alpha=0.5)
                
                x_min, x_max = ax_db.get_xlim()
                ax_db.set_xlim(x_min - (x_max-x_min)*0.25, x_max + (x_max-x_min)*0.25)
                ax_db.set_ylim(-1.0, len(risk_factors) + 0.7)
                ax_db.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False, fontsize=6)
                
                plt.tight_layout()
                st.pyplot(fig_db, clear_figure=True, use_container_width=False)

                st.markdown("#### 🎯 动作处方：")
                for _, row in risk_factors.iterrows():
                    f_name = row['feature']
                    advice = ADVICE_MAP.get(f_name, "保持良好姿势。")
                    st.info(f"👉 **{f_name}**: {advice}")
            else:
                st.success("✨ 您的动作表现非常平衡，未发现明显的力学风险项！")
            
            st.markdown("---")

    except Exception as e:
        st.error(f"🚨 分析执行出错: {e}")
        st.code(traceback.format_exc())
# ===================== 4. 运行逻辑 =====================
# ===================== 4. 运行逻辑 =====================
if st.button("🚀 开始自动化分析", use_container_width=True):
    if model_file is not None:
        run_analysis(session_id, trial_keyword, model_file)
    elif os.path.exists(DEFAULT_MODEL_NAME):
        run_analysis(session_id, trial_keyword, DEFAULT_MODEL_NAME)
    else:
        st.error(f"⚠️ 找不到模型文件！请上传 .pkl 文件或确保仓库中存在 {DEFAULT_MODEL_NAME}")

st.sidebar.markdown("---")
st.sidebar.caption("Powered by OpenCap & MLP Model")