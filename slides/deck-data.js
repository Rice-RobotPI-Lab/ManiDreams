window.DECK = [
  {
    "id": "title",
    "title": "ManiDreams",
    "duration": 15,
    "steps": 0,
    "stepLabels": [],
    "cn": "首页用中间两排演示 GIF 自动循环，简短自我介绍，不逐字念论文长标题。随后进入 world-model 提问。",
    "script": "I'm Gaotian Wang from Rice University. Today I'll present ManiDreams, our open-source library for robust object manipulation. Let me begin with a decision a robot has to make.",
    "bridge": "Suppose a world model gives you this prediction.",
    "sources": [
      "Paper p.1, title, affiliations and abstract.",
      "Selected GIF: out/promo-gifs/08_intro_and_code/title_full.gif. Middle two rows, crop 1440×388 at (0,440), complete 3-second loop."
    ]
  },
  {
    "id": "opening",
    "title": "Would you press execute?",
    "duration": 30,
    "steps": 1,
    "stepLabels": [
      "Let the audience consider the prediction",
      "Introduce the uncertain friction"
    ],
    "cn": "标题页一句话介绍后，先让观众看画面两秒，再问是否会按 Execute。第一次翻页只出现摩擦问题；这是一个假设，不把画面当成失败率证据。",
    "script": "A world model lets a robot rehearse an action before taking it. Suppose it predicts a successful push. Would you press execute?\n\n[Pause. Reveal the question.]\n\nNow change the friction. The imagined motion may look convincing, while the real outcome changes. The robot has to choose before it knows the exact physical conditions. How should a planner make that decision?",
    "bridge": "To make that decision, the planner has to account for several kinds of uncertainty.",
    "sources": [
      "Opening question is a hypothetical framing, not an experimental claim.",
      "Paper p.1, Introduction.",
      "Visual: supplied promotional video, 15.0–17.9 s, right prediction panel."
    ]
  },
  {
    "id": "uncertainty",
    "title": "Uncertainty reaches the planner",
    "duration": 25,
    "steps": 2,
    "stepLabels": [
      "Perception",
      "Physical parameters",
      "Model structure"
    ],
    "cn": "用三个具体问题解释，而不是念三种术语。观测错了、物理参数错了、接触模型也可能不够准确。不要说所有 world model 都只能预测单一未来。",
    "script": "Uncertainty enters in three ways. Perception: where is the object? Physical parameters: what are its mass and friction? Model structure: how well does the predictor describe the contact?\n\nThe planner has to consider these possibilities when selecting an action. We start by changing its representation of the object.",
    "bridge": "First: what could the object's state and physical context be?",
    "sources": [
      "Paper p.1, Introduction; pp.2–3, Sec.III.",
      "Visual badges extracted from the original project-page splash.svg; artwork retained without redrawing.",
      "Structural uncertainty motivates interchangeable TSIP backends; the quantitative benchmark tests observation noise, delay and physics perturbations."
    ]
  },
  {
    "id": "dris",
    "title": "One object, several plausible states",
    "duration": 40,
    "steps": 1,
    "stepLabels": [
      "Dream it: Single Instance / DRIS",
      "Each hypothesis pairs a state with a physical context"
    ],
    "cn": "三段对比的第一段，先让 GIF 完整播放一轮（4 秒）。左边是单实例，右边是同一目标的多种假设，不是场景里多了几个真实物体。介绍 DRIS 的 state/context 配对后就停，把如何预测留到下一页。",
    "script": "The first step is to represent uncertainty with a Domain-Randomized Instance Set, or DRIS.\n\n[Let the clip loop once.]\n\nThe left side has one instance. The right carries several plausible states and physical contexts for the same target. These are hypotheses about one object.\n\nThe formula pairs each state with a context, such as a choice of physical parameters. This set is what the planner carries forward.",
    "bridge": "Once we have plausible states, we need to predict their evolution under an action.",
    "sources": [
      "Paper p.3, Sec.III-B1, Eq.(1).",
      "Approved GIF: out/promo-gifs/01_comparisons/01_single_vs_dris.gif.",
      "Promotional video 9.0–13.0 s, lower comparison panel. Qualitative illustration."
    ]
  },
  {
    "id": "predict",
    "title": "Different actions, different possible outcomes",
    "duration": 30,
    "steps": 1,
    "stepLabels": [
      "Predict it: Drop / Push",
      "Ask which predicted spread the task should accept"
    ],
    "cn": "三段对比的第二段，播放完整 Drop / Push（4 秒）。用它说明预测要随动作传播整个 DRIS。不要说这两段是相同初始条件下的受控实验，也不要得出 push 比 drop 更好的结论。最后的问题自然引出 Cage。",
    "script": "Next, a Task-Specific Intuitive Physics model, or TSIP, takes a DRIS and an action and predicts the next DRIS.\n\n[Point to Drop, then Push.]\n\nThese clips illustrate how different actions change the spread of plausible object states. The prediction keeps uncertainty visible. But it does not tell us which outcomes satisfy the task. We need a way to judge that spread.",
    "bridge": "The cage gives the planner a task-specific way to judge those predicted outcomes.",
    "sources": [
      "Paper p.3, Sec.III-B2, Eq.(2): TSIP maps a DRIS and action to the next DRIS.",
      "Approved GIF: out/promo-gifs/01_comparisons/02_drop_vs_push.gif.",
      "Promotional video 15.0–19.0 s. Qualitative examples, not a controlled comparison of action quality."
    ]
  },
  {
    "id": "cage",
    "title": "Which outcomes does the task accept?",
    "duration": 35,
    "steps": 1,
    "stepLabels": [
      "Constrain it: Just Push / Caged",
      "Cage cost and containment check"
    ],
    "cn": "三段 bunny 对比的最后一段，先看完5秒动作。解释 evaluate 排序、validate 检查 containment。不要把动画称为安全保证。接着展示 teaser 概括方法，再讲分层框架，最后进入 loop。",
    "script": "The cage expresses what containment means for the task.\n\n[Let the comparison play.]\n\nIn this illustration, Just Push lets the plausible instances spread apart. The Caged behavior accounts for that spread when selecting actions.\n\nThe cage supplies a cost through evaluate and a containment check through validate. The Solver uses both to select an action. Those are the pieces we will now put together.",
    "bridge": "The bunny makes the idea concrete. Here is how it generalizes across tasks.",
    "sources": [
      "Paper p.3, Sec.III-B3; p.4, Algorithm 1.",
      "Approved GIF: out/promo-gifs/01_comparisons/03_just_push_vs_caged.gif.",
      "Promotional video 21.0–26.0 s. Qualitative illustration; no success-rate or safety-guarantee claim."
    ]
  },
  {
    "id": "teaser",
    "title": "Uncertainty inside the planning loop",
    "duration": 25,
    "steps": 0,
    "stepLabels": [],
    "cn": "bunny 后立即展示原来的 teaser 原图。先指中央 DRIS、TSIP 和 cage 的关系，再指周围任务与后端。用这张总览图将单个直观例子扩展到整篇工作的主题，不重复细讲三类不确定性。",
    "script": "The bunny demonstration captures the central idea in this figure. DRIS represents plausible object states and physical contexts. TSIP predicts their evolution, and the cage expresses what the task accepts.\n\nThe surrounding examples use different tasks and physics backends. ManiDreams integrates these components while keeping their interfaces separate.",
    "bridge": "The library organizes that separation into three layers.",
    "sources": [
      "Paper p.1, Fig.1 and abstract; pp.2–4, Sec.III.",
      "Original project-page splash.svg, previously on the title slide."
    ]
  },
  {
    "id": "architecture",
    "title": "A layered framework for manipulation",
    "duration": 25,
    "steps": 0,
    "stepLabels": [],
    "cn": "三层框架进入主讲，紧接 teaser。沿原图从底到顶讲：抽象接口 → 具体实现与 ManiDreamsEnv → 任务配置。说明组件如何解耦即可，把 sample/predict/constrain/execute 的运行细节留给下一页。",
    "script": "At the bottom are the abstract interfaces. The middle layer provides concrete physics models, executors, cages and solvers, connected through ManiDreamsEnv. The top layer supplies the task scene, pipeline configuration and sampler.\n\nThis separation lets a task change its predictor or executor while keeping the planning interface.",
    "bridge": "Now let's follow one planning cycle through those components.",
    "sources": [
      "Paper pp.2–4, Sec.III and Fig.2."
    ]
  },
  {
    "id": "loop",
    "title": "Together, they form a planning loop",
    "duration": 50,
    "steps": 3,
    "stepLabels": [
      "Sample candidate actions",
      "Predict distributional outcomes",
      "Constrain and select",
      "Execute a chunk and observe again"
    ],
    "cn": "前两页已经完成方法总览与软件分层，这里集中讲运行时。四步高亮依次讲 sample/predict/constrain/execute。保留原流程图和修正后的选框，不把 learned TSIP 描述成显式 N×m 个物理实例。",
    "script": "At runtime, the Solver connects these interfaces into a planning loop.\n\nFirst, sample candidate actions, potentially from an existing policy. TSIP then predicts a DRIS outcome for each candidate. Simulation can evaluate actions and instances in parallel; a learned backend predicts a representation of the DRIS.\n\nThe cage scores the outcomes and checks containment. The optimizer selects the best valid action and can refine candidates iteratively.\n\nWe execute a chunk, observe the real scene, refresh the DRIS and plan again. The policy still proposes actions. ManiDreams checks their distributional consequences before execution, without retraining the policy.",
    "bridge": "Because those interfaces are separate, we can reuse the loop across tasks.",
    "sources": [
      "Paper pp.4–5, Sec.IV, Algorithm 1 and Fig.3.",
      "Paper p.3: learned TSIP operates on a feature representation of the entire DRIS.",
      "Project-page ManiDreams_pipeline.svg."
    ]
  },
  {
    "id": "tasks",
    "title": "The same interfaces support different tasks",
    "duration": 25,
    "steps": 0,
    "stepLabels": [],
    "cn": "抓球强调 PPO 是 sampler；挑卡强调 3D cage + MPPI。每个例子讲一句，不逐项讲所有实现。",
    "script": "For ball catching, a pretrained PPO policy supplies candidate actions. A simulation TSIP predicts their effects, and the Solver selects an action.\n\nCard picking uses a three-dimensional geometric cage, continuous action sampling and MPPI refinement to move the card toward a pickable configuration.\n\nThe planning interface stays the same while the task components change.",
    "bridge": "We can also hold the task fixed and change the physics backend.",
    "sources": [
      "Paper pp.5–6, Sec.VI-A, Figs.6–7.",
      "Project-page release media-v1: 13.mp4 and 14.mp4."
    ]
  },
  {
    "id": "backends",
    "title": "Two physics backends, the same task",
    "duration": 30,
    "steps": 0,
    "stepLabels": [],
    "cn": "先指 simulation，再指 diffusion。这里真正要观众记住的是 same cage, same sampler, same solver，而不是“学出来一定更好”。",
    "script": "Here, both examples perform circular object pushing. One uses simulation, and the other uses a diffusion-based predictor. They share the cage, discrete action sampler and N-best solver.\n\nThis is the integration idea in software: changing the predictor does not require rewriting the constraint or action-selection logic. The simulation predicts physical instances; the learned backend operates on a DRIS representation.",
    "bridge": "The same separation lets us replace the executor with a physical robot.",
    "sources": [
      "Paper p.5, Sec.VI-A, Pushing; Fig.5.",
      "Project-page release media-v1: 11.mp4 and 12.mp4. Plan and execute views placed side by side."
    ]
  },
  {
    "id": "robot",
    "title": "The loop reaches the physical robot",
    "duration": 45,
    "steps": 1,
    "stepLabels": [
      "Four targets across two tasks",
      "Chunked execution and perception"
    ],
    "cn": "按左右两列讲四个目标：左列 croissant、chicken leg，先推开再抓；右列 steak、thin box，沿墙角铲取。点出同类任务中的不同形状即可，不逐帧解说。SAM2 来自透明桌面下方相机，原片 x2 标识保留。四段是真机定性演示，不代表四项成功率实验。页面仍讲45秒。",
    "script": "On a Franka Panda, we swap in the real-world executor without fine-tuning. A camera beneath the transparent table supplies SAM2 masks as the pixel DRIS, and a diffusion TSIP predicts its evolution.\n\nOn the left, croissant and chicken leg show push-picking: create clearance in clutter, then grasp. On the right, steak and thin box show scooping: push a flat target to a wall corner and use the geometry to lift it.\n\nAll four run in chunks of eight actions, refreshing perception between chunks. These are qualitative deployment demonstrations.",
    "bridge": "For a controlled comparison, we return to the ManiSkill benchmarks.",
    "sources": [
      "Paper pp.6–7, Sec.VI-B and Fig.8.",
      "Release media-v1: real1.mp4, top-left croissant and top-right chicken leg; real2.mp4, bottom-left steak and top-left thin box.",
      "All four objects appear in paper Fig.8. Complete 20-second source trials and x2 speed labels are retained."
    ]
  },
  {
    "id": "benchmark",
    "title": "The same PPO policy under perturbation",
    "duration": 65,
    "steps": 1,
    "stepLabels": [
      "Nine task–perturbation combinations",
      "Enlarge PushT under observation delay"
    ],
    "cn": "先交代公平性：同一个 PPO，直接执行 vs 作为 sampler，无重训。不要逐条曲线念；指向中间 delay 列，再总结中等扰动收益最大、极端扰动仍会退化。",
    "script": "Here, the baseline executes a PPO policy directly. ManiDreams uses that same PPO policy as its sampler, without retraining it. We test PushCube, PickCube and PushT, shown from top to bottom.\n\nThe columns vary observation noise, observation delay and physics perturbations separately, with the other two held at zero. ManiDreams uses eight action samples and eight DRIS instances. Each configuration has ten runs of one hundred episodes.\n\nThe main pattern is that ManiDreams generally outperforms the baseline across the nine task and perturbation combinations. The gap is most pronounced at moderate perturbations. For example, here is PushT under increasing observation delay.\n\nAt extreme perturbations, the fixed DRIS can stop covering the uncertainty. Both methods then degrade. The representation's coverage remains a practical limit.",
    "bridge": "How much coverage do we need, and what does it cost?",
    "sources": [
      "Paper pp.6–7, Sec.VI-C and Fig.9. Shading: ±1 standard deviation.",
      "N=8, m=8, cage weight λ=0.1. The three perturbations are swept independently.",
      "Project-page Exp_1.svg. No values estimated from curves."
    ]
  },
  {
    "id": "tradeoff",
    "title": "Coverage and compute set the trade-off",
    "duration": 65,
    "steps": 1,
    "stepLabels": [
      "DRIS size, candidates and width",
      "Runtime measurements"
    ],
    "cn": "默认 m=N=8, Medium 的 82% 只在该 ablation 配置成立。宽度窄74/中82/宽76不是跨任务最优。延迟图的129.2ms必须保留，避免说所有配置10–20Hz。",
    "script": "On PushCube, increasing the DRIS size and the number of candidate actions improves success, with diminishing returns beyond sixteen in this experiment. Distribution width matters as well. Narrow, medium and wide settings achieve seventy-four, eighty-two and seventy-six percent. More uncertainty coverage is not automatically better if it makes the planner overly conservative.\n\nThe right side shows the measured cost on one RTX 5070 Ti laptop GPU. PPO alone takes twenty point two milliseconds per step. The simulation configurations range from forty-nine point six to one hundred twenty-nine point two milliseconds. The diffusion TSIP takes fifty-five point two.\n\nThese measurements expose a tunable trade-off. They do not remove the need to choose a suitable DRIS, cage and physics backend for the task.",
    "bridge": "That brings us back to the decision in the opening.",
    "sources": [
      "Paper p.7, Table I and Sec.VI-D; pp.7–8, Fig.10 and Sec.VI-E.",
      "Ablation: noise σ=0.04 m, chunk length 3, physics severity 0.4, 100 episodes/config.; vary one parameter with others at m=N=8, Medium.",
      "The 64×8 runtime is 129.2 ms in Fig.10; do not claim all configurations run at 10–20 Hz."
    ]
  },
  {
    "id": "closing",
    "title": "Which futures should the robot accept?",
    "duration": 35,
    "steps": 0,
    "stepLabels": [],
    "cn": "回扣开场：从“是否相信预测”落到“用分布约束选动作”。说完核心句后停顿，再指二维码。结束停留此页；备份页不计入9分钟。",
    "script": "At the beginning, I asked whether you would press execute after seeing a convincing prediction. ManiDreams makes the decision more explicit: represent plausible worlds, predict their outcomes, and choose actions whose spread satisfies the task's cage.\n\nThe contribution is an open-source integration of those pieces, around an existing policy. The current limits include manual DRIS and cage tuning, and the need for a suitable dynamics backend.\n\nThe code, examples and documentation are available here. Thank you.",
    "bridge": "",
    "sources": [
      "Paper p.8, Conclusion and limitations.",
      "Project: https://rice-robotpi-lab.github.io/ManiDreams/",
      "Repository: https://github.com/Rice-RobotPI-Lab/ManiDreams"
    ]
  },
  {
    "id": "configuration",
    "title": "Backup: evaluation settings",
    "duration": 0,
    "steps": 0,
    "stepLabels": [],
    "cn": "应对 benchmark是否同时加噪、ablation范围以及baseline公平性的问题。root paper 没有说 PPO 训练是否使用 domain randomization，不添加这个结论。",
    "script": "The robustness benchmark varies one perturbation at a time, with the other two at zero. In contrast, the ablation combines observation noise, chunk length and physics perturbation at fixed levels while varying a single design parameter. Narrow, Medium and Wide multiply default physical parameters by the ranges shown. The baseline uses the same PPO policy and bypasses TSIP evaluation with one sample and no cage.",
    "bridge": "",
    "sources": [
      "Paper pp.6–7, Sec.VI-C/D, Table I; p.5, Sec.V."
    ]
  },
  {
    "id": "limitations",
    "title": "Backup: scope and limitations",
    "duration": 0,
    "steps": 0,
    "stepLabels": [],
    "cn": "明确区分架构目标、定量结果与定性真机演示。不要声称 arbitrary uncertainty 下的安全保证，也不要把演示视频当作统计评测。",
    "script": "ManiDreams exposes uncertainty through a shared planning interface. Its quantitative experiments support improved robustness to the tested perceptual and parametric perturbations. Backend interchangeability accommodates different modeling choices, but the benchmark does not establish a general guarantee against arbitrary structural model error. DRIS parameters and cages still need task-specific tuning. Highly discontinuous dynamics may require a more expressive TSIP. The real-robot section reports demonstrations rather than a statistical success-rate study.",
    "bridge": "",
    "sources": [
      "Paper pp.6–8, Sec.VI and Conclusion."
    ]
  }
];
