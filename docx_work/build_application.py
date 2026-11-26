from __future__ import annotations

import os
from pathlib import Path
from zipfile import ZipFile

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt


REFERENCE = Path(r"C:\Users\Admin\Desktop\for now\大创申报书.docx")
OUTPUT = Path(
    r"D:\大学编程学习\rl+transformer+具身智能\attm5\docx_work\CELAR_大创申报书_试填稿.docx"
)
FIGURE_DIR = OUTPUT.parent / "figures"

PROJECT_NAME = "异构机械臂跨本体动作嵌入与强化学习"

ABSTRACT = (
    "异构机械臂的动作维度、运动学与夹爪差异使策略难以复用，新本体适配依赖大量演示。"
    "本项目学习由RGB-D、语言、本体状态和动作块构成的共享潜在动作，以本体条件解码器和"
    "安全层生成可执行控制，并联合跨本体对齐、循环与后果一致约束。前期已完成三本体配对"
    "仿真、1,200条RGB-D轨迹和工程验证，并获得行为克隆、PPO及少样本适配的正向证据。"
    "本项目将开展多任务在线训练与外部验证，评估成功率、样本效率和安全性。"
)

INNOVATION_SHORT = (
    "共享潜在动作由本体条件decoder和安全层约束，避免直接拼接异构关节；联合对齐、循环"
    "与后果一致训练，兼顾语义共享和可执行性；将新本体适配拆为decoder少样本学习与高层"
    "策略复用。"
)


def set_run_font(run, *, size: float = 10.5, bold: bool = False) -> None:
    run.font.name = "宋体"
    run.font.size = Pt(size)
    run.bold = bold
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")


def replace_in_cell(table, row: int, column: int, replacements: list[tuple[str, str]]) -> None:
    for paragraph in table.cell(row, column).paragraphs:
        for run in paragraph.runs:
            for old, new in replacements:
                if old in run.text:
                    run.text = run.text.replace(old, new)


def clear_cell(cell):
    first = cell.paragraphs[0]
    for paragraph in cell.paragraphs[1:]:
        paragraph._element.getparent().remove(paragraph._element)
    for run in list(first.runs):
        first._element.remove(run._element)
    first.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return first


def add_paragraph(cell, text: str, *, first=None, bold_lead: str | None = None, size=10.5):
    paragraph = first if first is not None else cell.add_paragraph()
    paragraph.style = cell.part.document.styles["Normal"] if hasattr(cell.part, "document") else paragraph.style
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(3)
    paragraph.paragraph_format.line_spacing = 1.25
    paragraph.paragraph_format.first_line_indent = Pt(21)
    if bold_lead and text.startswith(bold_lead):
        lead = paragraph.add_run(bold_lead)
        set_run_font(lead, size=size, bold=True)
        remainder = paragraph.add_run(text[len(bold_lead) :])
        set_run_font(remainder, size=size)
    else:
        run = paragraph.add_run(text)
        set_run_font(run, size=size)
    return paragraph


def fill_cell(cell, paragraphs: list[tuple[str, str | None]]) -> None:
    first = clear_cell(cell)
    for index, (text, lead) in enumerate(paragraphs):
        add_paragraph(
            cell,
            text,
            first=first if index == 0 else None,
            bold_lead=lead,
        )


def insert_choice(cell, text: str) -> None:
    paragraph = cell.paragraphs[0].insert_paragraph_before()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(2)
    run = paragraph.add_run(text)
    set_run_font(run, size=10.5, bold=True)


def keep_row_together(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    if tr_pr.find(qn("w:cantSplit")) is None:
        tr_pr.append(OxmlElement("w:cantSplit"))


def add_figure(
    cell,
    image_path: Path,
    caption: str,
    *,
    width_inches: float = 5.45,
) -> None:
    paragraph = cell.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(6)
    paragraph.paragraph_format.space_after = Pt(3)
    paragraph.paragraph_format.keep_with_next = True
    paragraph.paragraph_format.keep_together = True
    paragraph.add_run().add_picture(str(image_path), width=Inches(width_inches))
    caption_paragraph = cell.add_paragraph()
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_paragraph.paragraph_format.space_before = Pt(0)
    caption_paragraph.paragraph_format.space_after = Pt(7)
    caption_paragraph.paragraph_format.first_line_indent = Pt(0)
    caption_paragraph.paragraph_format.keep_together = True
    run = caption_paragraph.add_run(caption)
    set_run_font(run, size=8.5)
    run.italic = True


doc = Document(REFERENCE)

# Cover project name
cover = doc.paragraphs[4]
cover.runs[1].text = PROJECT_NAME

# Project metadata
table0 = doc.tables[0]
set_project = clear_cell(table0.cell(0, 1))
add_paragraph(set_project, PROJECT_NAME, first=set_project)
insert_choice(table0.cell(1, 1), "勾选：☑面上项目")
insert_choice(table0.cell(2, 1), "勾选：☑科学探索与工程技术类")
set_subject = clear_cell(table0.cell(4, 1))
add_paragraph(set_subject, "计算机类（0809）", first=set_subject)
insert_choice(
    table0.cell(6, 1),
    "勾选：☑基于前期研究实践成果、继续深入研究实践的创新项目",
)
insert_choice(table0.cell(9, 1), "选择：☑E.人工智能")

# Abstract and innovation points
table2 = doc.tables[2]
fill_cell(table2.cell(1, 0), [(ABSTRACT, None)])
clear_cell(table2.cell(2, 0))
clear_cell(table2.cell(3, 0))
fill_cell(table2.cell(5, 0), [(INNOVATION_SHORT, None)])
table2.cell(4, 0).paragraphs[0].paragraph_format.page_break_before = True
for row_index in (4, 7, 9):
    keep_row_together(doc.tables[3].rows[row_index])

rationale = [
    (
        "1. 研究背景与意义。机械臂已进入物流分拣、柔性装配、实验室自动化和家庭服务等场景，"
        "但不同厂商在关节数量、连杆尺度、控制频率、夹爪结构和动力学参数上差异显著。现有策略通常"
        "针对单一机械臂采集数据与训练，换用新本体后需要重新采集大量演示并做在线调参，造成数据、"
        "算力和部署成本重复投入。研究一种能够跨机械臂共享、又能落地为可执行控制的动作表示，对于"
        "降低异构机器人系统的适配成本具有直接意义。",
        "1. 研究背景与意义。",
    ),
    (
        "2. 国内外研究现状。RT-1、Open X-Embodiment、Octo和π0等工作推动了多机器人数据与视觉"
        "语言动作模型的发展，说明大规模异构数据能够改善泛化；但这类方法依赖大模型与大算力，且通常将"
        "不同机器人数据混合训练。LAPA从无动作视频学习潜在动作并迁移到机器人控制，CLAM研究无标签"
        "演示中的连续潜在动作，Latent Action Diffusion和XSkill进一步探索跨本体技能共享；VQ-BeT说明"
        "离散潜在动作可用于策略生成。AnyBody提供了跨本体操作评测基础，HPT强调本体感知条件化和异构"
        "预训练，跨本体统一策略的实证研究也揭示了数据规模与动作空间差异的双重影响。对于固定视觉主干"
        "和中小规模数据预算，仍需回答动作空间如何共享、共享表示如何保持可执行性以及能否转化为强化"
        "学习样本效率等问题。",
        "2. 国内外研究现状。",
    ),
    (
        "3. 亟待解决的问题。第一，不同本体的原生动作向量维度不同，直接拼接会混入无意义维度；"
        "第二，轨迹虽可在任务阶段上配对，但低层运动细节并不相同，过度对齐可能损害控制精度；"
        "第三，潜在动作作为策略输出时必须经过本体解码器、关节限位和速度约束，否则仿真收益不能转化为"
        "可执行动作；第四，新本体适配需要区分动作接口迁移与高层策略迁移，避免把 decoder 的可执行性"
        "误写成任务成功率。",
        "3. 亟待解决的问题。",
    ),
    (
        "4. 前期基础。项目组已完成跨本体动作表示原型，并在Panda、UR5e和Sawyer上建立最小统一"
        "仿真环境。当前版本生成抓取、放置、开抽屉和推物体4类任务，每本体每任务100条配对轨迹，"
        "共1,200条128×128 RGB-D轨迹；配对、任务和场景切分无泄漏，初始状态可重复恢复，专家动作经"
        "安全层后关节与速度越界均为0。受控代理实验显示：相似动作检索R@1达到0.911，普通共享自动"
        "编码器为0.550；CELAR PPO最终成功率为0.715，原生动作PPO为0.375；未见本体5条演示下"
        "decoder语义误差降低26.0%。这些结果用于支持下一步真实接触动力学验证，不作为最终论文结论。",
        "4. 前期基础。",
    ),
    (
        "5. 预期价值。本项目拟形成一套面向异构机械臂的共享动作表示、可执行解码与强化学习流程，"
        "并建立可复现的配对数据、训练代码和评测协议。若阶段门通过，可为多机器人软件复用、少量演示"
        "快速部署和跨本体策略迁移提供技术基础；若不通过，也将明确共享动作瓶颈、策略条件化或接触建模"
        "中的具体失效环节，避免将无法复现的泛化能力作为项目成果。",
        "5. 预期价值。",
    ),
    (
        "主要参考文献：\n"
        "[1] Latent Action Pretraining from Videos. arXiv:2410.11758.\n"
        "[2] CLAM: Continuous Latent Action Models for Robot Learning from Unlabeled "
        "Demonstrations. arXiv:2505.04999.\n"
        "[3] Latent Action Diffusion for Cross-Embodiment Manipulation. arXiv:2506.14608.\n"
        "[4] Cross-Embodiment Robot Manipulation Skill Transfer using Latent Space Alignment. "
        "arXiv:2406.01968.\n"
        "[5] XSkill: Cross Embodiment Skill Discovery. arXiv:2307.09955.\n"
        "[6] Behavior Generation with Latent Actions. arXiv:2403.03181.\n"
        "[7] AnyBody: A Benchmark Suite for Cross-Embodiment Manipulation. arXiv:2505.14986.\n"
        "[8] Pushing the Limits of Cross-Embodiment Learning for Manipulation and Navigation. "
        "arXiv:2402.19432.\n"
        "[9] Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers. "
        "arXiv:2409.20537.\n"
        "[10] RT-1: Robotics Transformer for Real-World Control at Scale. arXiv:2212.06817.\n"
        "[11] Open X-Embodiment: Robotic Learning Datasets and RT-X Models. arXiv:2310.08864.",
        None,
    ),
]

fill_cell(doc.tables[3].cell(2, 0), rationale[:-1])
add_figure(
    doc.tables[3].cell(2, 0),
    FIGURE_DIR / "alignment_rl_results.png",
    "图1  C1 跨本体动作检索与 C3 等预算 PPO。柱状图为 3 个随机种子的均值及 95% 置信区间；"
    "PPO 曲线使用相同 49,152 个在线环境步。",
    width_inches=4.9,
)
add_figure(
    doc.tables[3].cell(2, 0),
    FIGURE_DIR / "group_adaptation_results.png",
    "图2  C3 本体-任务分组与 C4 未见 Sawyer 适配。左图按本体和任务汇总 3 个种子；"
    "右图比较 5/20/50 条演示下的动作语义误差，50-shot 反向结果提示收益主要集中在极少样本区间。",
    width_inches=5.2,
)
add_paragraph(doc.tables[3].cell(2, 0), rationale[-1][0])
doc.tables[3].cell(4, 0).paragraphs[1].paragraph_format.page_break_before = True

research_content = [
    (
        "关键科学问题。问题一：在动作维度与动力学不一致的条件下，如何学习稳定的跨本体动作语义；"
        "问题二：共享潜在动作如何经过本体条件解码后持续满足关节限位、速度、夹爪和工作空间约束；"
        "问题三：共享动作空间能否在相同在线预算下改善强化学习成功率，并支持新本体少量演示适配。",
        "关键科学问题。",
    ),
    (
        "研究内容一：统一配对数据与本体描述。围绕抓取、放置、开抽屉和推物体建立共享任务ID、场景ID、"
        "初始状态和目标，采集RGB-D、本体状态、原生动作、末端位姿、夹爪、进度、成功与安全约束。"
        "保留失败和扰动轨迹，并以pair、task instance和scene为分组单位切分训练、验证和测试集。",
        "研究内容一：统一配对数据与本体描述。",
    ),
    (
        "研究内容二：跨本体潜在动作模型。以轻量视觉编码器提取RGB-D特征，联合语言、本体状态和本体"
        "描述形成上下文；动作编码器读取完整动作块并输出8至16维潜在动作。各本体设置独立轻量decoder，"
        "通过重建、跨本体对齐、循环一致、任务进度后果一致和平滑损失联合训练。重点监测潜在变量使用率、"
        "有效秩和跨本体检索，防止模型通过常量输出获得低重建误差。",
        "研究内容二：跨本体潜在动作模型。",
    ),
    (
        "研究内容三：可执行解码与安全层。decoder输出经过关节限位、最大单步变化、速度、夹爪开度和"
        "工作空间检查；动作块在环境中完整执行并记录安全触发次数。对接触阶段增加短块约束和扰动测试，"
        "分析潜在动作方向变化与任务进度变化是否一致。",
        "研究内容三：可执行解码与安全层。",
    ),
    (
        "研究内容四：行为克隆与在线强化学习。先以行为克隆验证同一本体内控制接口，再冻结decoder，"
        "使用PPO在潜在动作空间训练高层策略。与原生动作PPO、普通共享decoder和每本体独立策略在相同"
        "种子、环境数量、在线步数和奖励函数下比较，记录达到50%与80%成功率的步数。",
        "研究内容四：行为克隆与在线强化学习。",
    ),
    (
        "研究内容五：新本体适配、多任务和外部验证。Sawyer作为预训练阶段未见本体，比较5、20、50条"
        "演示下的decoder适配、独立BC和共享策略方案；随后扩展到四类操作任务，并在新物体、初始位姿、"
        "相机和语言改写上测试。外部验证至少选择一个公开机器人基准，检查结论是否只存在于自建环境。",
        "研究内容五：新本体适配、多任务和外部验证。",
    ),
    (
        "总体框架。数据层统一三本体四任务的配对轨迹与安全标签；表示层学习共享潜在动作并为每本体保留"
        "独立decoder；策略层在潜在空间进行BC和PPO；评测层同时报告动作可执行率、语义检索、任务成功率、"
        "样本效率和新本体负迁移。各阶段设置预注册门控，未通过时不进入下一阶段。",
        "总体框架。",
    ),
    (
        "主要目标。完成三本体四任务的数据与环境门槛；验证跨本体动作语义对齐和可执行解码；证明同等"
        "在线预算下的成功率或样本效率收益；验证5至20条演示的新本体适配；形成可复现实验包、技术报告"
        "和阶段性论文材料。阶段性指标为动作检索明显高于共享AE，BC语义误差不高于原生BC的115%，"
        "在线成功率高5个百分点或达标步数少30%，多数本体-任务组合方向为正。",
        "主要目标。",
    ),
]

fill_cell(doc.tables[3].cell(5, 0), research_content)

technical_plan = [
    (
        "1. 数据与环境。使用MuJoCo/Robosuite统一相机、桌子、目标区域和四类物体，构建Panda、UR5e、"
        "Sawyer的配对场景。同一初始状态在三本体上分别执行专家轨迹，动作块长度设为4、6、8，"
        "并记录完整块执行后的观测、姿态、进度和约束状态。对每类任务保留nominal、failure与perturb"
        "轨迹，固定随机种子并保存配置、源码与数据哈希。",
        "1. 数据与环境。",
    ),
    (
        "2. 表示学习。动作编码器采用小型Transformer或MLP，输入原生动作块、当前与下一时刻上下文和"
        "本体描述；潜在维度统一为8至16。训练目标包含动作重建、同任务同阶段配对对齐、A本体潜在动作"
        "经B本体decoder再编码的循环一致、不同本体任务进度变化一致、时间平滑和本体条件约束。"
        "以普通共享AE、独立AE、去掉对齐、去掉循环、打乱配对等作为对照。",
        "2. 表示学习。",
    ),
    (
        "3. decoder与安全。每个本体使用1至3M参数的独立decoder，输出本体原生长度动作块。"
        "安全层按关节顺序逐维裁剪，限制单步变化和工作空间，并在夹爪与接触阶段记录触发原因。"
        "安全层不参与梯度回传，防止策略通过放宽约束获得表面收益。",
        "3. decoder与安全。",
    ),
    (
        "4. BC与RL。先训练上下文到原生动作和上下文到潜在动作的行为克隆，使用相同验证集和早停规则。"
        "RL阶段冻结动作编码器和decoder，使用PPO更新高层策略；奖励由任务成功、任务进度、抓取或接触、"
        "碰撞、限位和动作抖动组成。所有方法使用相同环境步数、并行环境数、随机种子和评估回合。",
        "4. BC与RL。",
    ),
    (
        "5. 适配和评测。新本体只更新decoder或小型adapter，冻结共享表示与高层策略；分别报告teacher "
        "latent decoder与冻结策略闭环，避免混淆动作接口和任务策略。主指标包括动作重建误差、安全可执行率、"
        "相似动作R@1/R5、达到50%和80%成功率的步数、最终成功率、负迁移率和推理延迟，并给出多种子置信区间。",
        "5. 适配和评测。",
    ),
    (
        "6. 可行性。项目组已完成统一环境原型、1,200条配对轨迹生成、RGB-D与末端位姿对齐检查、"
        "动作块安全执行和显存/延迟测试。当前128×128 RGB-D、batch 8的预检峰值显存为0.260GB，"
        "三本体动作块执行中位数低于1ms；12GB GPU可支持先冻结视觉主干、再分阶段训练。"
        "若接触动力学带来性能下降，将优先采用RGB-D特征缓存、batch 4梯度累积和冻结视觉主干。",
        "6. 可行性。",
    ),
    (
        "7. 风险与回退。若跨本体对齐导致重建质量下降，则降低对齐权重并增加本体条件；若潜在表示塌缩，"
        "则采用方差、使用率和重构联合正则；若decoder误差被策略利用，则缩短动作块并加入反事实执行；"
        "若新本体负迁移，则减少共享层并增加adapter容量；若12GB显存不足，则先训练动作表示，再串行训练策略。",
        "7. 风险与回退。",
    ),
]

fill_cell(doc.tables[3].cell(8, 0), technical_plan)

innovation = [
    (
        "创新点一：面向异构动作空间的可执行共享表示。不是把Panda、UR5e、Sawyer的关节向量直接拼接，"
        "而是学习低维共享动作语义，并保留本体条件decoder和安全层，使跨本体共享与单本体可执行性同时成立。",
        "创新点一：面向异构动作空间的可执行共享表示。",
    ),
    (
        "创新点二：语义对齐与后果一致联合约束。在动作重建之外加入跨本体阶段配对、循环一致、任务进度"
        "后果一致和平滑约束，使潜在空间不仅可压缩，还能表达不同本体的相似任务作用。",
        "创新点二：语义对齐与后果一致联合约束。",
    ),
    (
        "创新点三：分层的少样本适配与严格阶段门。将新本体适配拆分为动作接口适配和高层策略复用，"
        "并用数据、安全、BC、样本效率、适配和外部验证阶段门逐级判断价值，保留失败轨迹和负迁移结果。",
        "创新点三：分层的少样本适配与严格阶段门。",
    ),
]

fill_cell(doc.tables[3].cell(10, 0), innovation)

foundation = [
    (
        "负责人及项目组已完成CELAR预研：搭建Panda、UR5e、Sawyer的统一MuJoCo环境，覆盖抓取、放置、"
        "开抽屉和推物体四类任务，生成1,200条128×128 RGB-D配对轨迹，建立原生动作、末端位姿、夹爪、"
        "进度、成功和安全约束字段。受控实验完成跨本体动作检索、潜在行为克隆、等预算PPO和新本体"
        "decoder适配，并保存逐样本数据、随机种子、源码哈希与校验和。上述工作证明团队具备数据处理、"
        "模型训练、仿真评测和结果复现能力。",
        "负责人及项目组已完成",
    )
]

fill_cell(doc.tables[4].cell(2, 0), foundation)
add_figure(
    doc.tables[4].cell(2, 0),
    FIGURE_DIR / "c0_rgbd_examples.png",
    "图3  三本体 128×128 RGB-D 样例。训练前工程检查中，RGB 与末端位姿投影误差为 "
    "0.66–0.83 像素，深度误差为 0.018 m。",
)

# Teacher research foundation remains blank for the adviser.
fill_cell(
    doc.tables[4].cell(6, 0),
    [
        (
            "项目组已具备代码开发与数据分析条件，可在Linux服务器上运行MuJoCo/Robosuite和PyTorch训练；"
            "当前实验环境为RTX 3060 12GB GPU、Python 3.10、PyTorch与128×128 RGB-D渲染链路。"
            "项目采用固定随机种子、配置快照、数据校验和和失败轨迹保留机制，能够支持多任务实验、"
            "阶段门审计和结果复现。",
            "项目组已具备",
        )
    ],
)

expected = doc.tables[8].cell(1, 0)
add_paragraph(
    expected,
    "拟选择：会议论文1篇；创新创业类竞赛；其他：可复现仿真环境、配对数据与代码1套"
    "（数量待确认）。",
)
doc.tables[8].cell(3, 0).paragraphs[0].paragraph_format.page_break_before = True

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
edited_only = OUTPUT.with_suffix(".edited-only.docx")
doc.save(edited_only)

# Preserve the retained template package byte-for-byte except for the body
# document part. python-docx may otherwise rewrite styles, numbering, core
# properties, and relationship parts even when only table text changed.
repacked = OUTPUT.with_suffix(".repacked.docx")
with ZipFile(REFERENCE, "r") as source, ZipFile(edited_only, "r") as edited, ZipFile(
    repacked, "w"
) as target:
    editable_parts = {
        "[Content_Types].xml",
        "word/document.xml",
        "word/_rels/document.xml.rels",
    }
    for info in source.infolist():
        payload = (
            edited.read(info.filename)
            if info.filename in editable_parts
            else source.read(info)
        )
        target.writestr(info, payload)
    source_names = {info.filename for info in source.infolist()}
    for info in edited.infolist():
        if info.filename not in source_names:
            target.writestr(info, edited.read(info))
os.replace(repacked, OUTPUT)
edited_only.unlink(missing_ok=True)
print(OUTPUT)
print("abstract_chars", len(ABSTRACT))
print("innovation_chars", len(INNOVATION_SHORT))
