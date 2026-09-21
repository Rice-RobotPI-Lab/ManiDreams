(() => {
"use strict";
const slides = window.DECK;
const MAIN = slides.filter(s => s.duration > 0).length;
const TOTAL = slides.reduce((n,s)=>n+s.duration,0);
const params = new URLSearchParams(location.search);
const isPresenter = params.has("presenter");
const isPreview = params.has("preview");
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const esc = s => String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const math = tex => '<span class="tex" data-tex="'+esc(tex)+'">'+katex.renderToString(tex,{throwOnError:true,strict:"error",output:"htmlAndMathml"})+'</span>';
const benchmarkCaption = zoom => (zoom ? "PushT under observation delay (enlarged from Fig. 9). " : "Rows: PushCube, PickCube, PushT. ")+math("N = m = 8")+(zoom ? "; " : ". Each perturbation swept independently. ")+math("10")+" runs "+math("\\times100")+" episodes; "+(zoom?"shading ":"")+math("\\pm 1")+" SD.";
const fig = (name,alt,cls="") => '<img class="figure '+cls+'" src="assets/figures/'+name+'.svg" alt="'+alt+'" draggable="false">';
const video = (name,label) => '<div class="video-wrap"><img class="video motion-media" src="assets/posters/'+name+'.jpg" data-gif="assets/gif/'+name+'.gif" data-poster="assets/posters/'+name+'.jpg" data-video="assets/video/'+name+'.mp4" alt="'+label+'" draggable="false"></div>';
const source = s => '<p class="evidence-caption">'+s+'</p>';
const robotExample = (name,object,description) => '<figure class="robot-example"><figcaption class="robot-label">'+object+'</figcaption>'+video(name,description)+'</figure>';
const title = (heading,sub="") => '<h2 class="slide-title">'+heading+'</h2>'+(sub?'<p class="lede">'+sub+'</p>':"");
const templates = {
opening:()=>'<div class="opening-copy"><p class="prelude">Your world model predicts<br>a successful push.</p><h1>Would you<br>press execute?</h1><p class="question fragment" data-step="1">Now change the friction.</p><p class="opening-foot fragment" data-step="1">The robot must act before it knows.</p></div><figure>'+video("hook","Plausible pushing outcomes in simulation")+source("Illustrative simulation from the project video")+'</figure>',
title:()=>'<div class="title-header"><h1 class="wordmark">ManiDreams</h1><img class="rice-mark" src="assets/figures/rice.svg" alt="Rice University"></div><p class="paper-title">An Open-Source Library for Robust Object Manipulation<br>via Uncertainty-aware Task-specific Intuitive Physics</p><div class="title-montage">'+video("title-montage","Two rows of manipulation examples from the original promotional title sequence")+'</div><p class="authors">Gaotian Wang¹ &nbsp; Kejia Ren¹ &nbsp; Andrew S. Morgan² &nbsp; Kaiyu Hang¹</p><p class="affiliations">¹ Rice University &nbsp;&nbsp; ² Robotics and AI Institute</p>',
teaser:()=>title("Uncertainty inside the planning loop","A common representation connects different tasks and physics backends.")+'<div class="teaser-figure">'+fig("splash","Original teaser: perceptual, structural and parametric uncertainty integrated through DRIS, TSIP and cage constraints")+'</div><p class="thesis">Robust manipulation is an integration problem.</p>',
uncertainty:()=>title("Uncertainty reaches the planner","A real action has to work with imperfect information.")+'<div class="uncertainty-columns"><div class="uncertainty-col">'+fig("uncertainty-perceptual","Perceptual uncertainty: the original eye and contour badge","uncertainty-badge")+'<h3>Perceptual</h3><p>Where is the object<br>right now?<span class="small">Noisy or delayed observations</span></p></div><div class="uncertainty-col fragment" data-step="1">'+fig("uncertainty-parametric","Parametric uncertainty: the original gear and plus-minus badge","uncertainty-badge")+'<h3>Parametric</h3><p>What physics will<br>this contact have?<span class="small">Mass, friction and geometry</span></p></div><div class="uncertainty-col fragment" data-step="2">'+fig("uncertainty-structural","Structural uncertainty: the original diverging trajectories badge","uncertainty-badge")+'<h3>Structural</h3><p>How well does the<br>model describe it?<span class="small">Imperfect dynamics models</span></p></div></div><p class="uncertainty-takeaway fragment" data-step="2">Carry plausible outcomes into action selection.</p>',
dris:()=>title("One object, several plausible states","Dream it · Domain-Randomized Instance Set (DRIS)")+'<div class="comparison-media">'+video("dris","Single instance on the left; domain-randomized instance set on the right")+'</div><div class="dris-bottom"><p>Multiple hypotheses about <strong>one target</strong>.<br>Different states and physical contexts.</p><div class="equation fragment" data-step="1">'+math("\\mathcal{D}_t = \\bigl\\{(s_t^{(i)}, c^{(i)})\\bigr\\}_{i=1}^{m}")+'<small>state '+math("s")+' &nbsp; / &nbsp; physical context '+math("c")+' &nbsp; / &nbsp; '+math("m")+' instances</small></div></div>',
predict:()=>title("Different actions, different possible outcomes","Predict it · Task-specific intuitive physics (TSIP)")+'<div class="comparison-media">'+video("predict","Drop on the left, Push on the right: qualitative examples of action-dependent DRIS evolution")+'</div><div class="comparison-bottom"><p>TSIP predicts the next DRIS<br>from the current set and an action.</p><p class="comparison-question fragment" data-step="1">Which predicted spread<br>should the task accept?</p></div>',
cage:()=>title("Which outcomes does the task accept?","Constrain it · Cage evaluation and validation")+'<div class="comparison-media">'+video("cage","Qualitative Just Push versus Caged comparison from the promotional video")+'</div><div class="cage-notes"><p class="fragment" data-step="1"><strong class="blue">evaluate()</strong> scores the outcome.<br><strong class="blue">validate()</strong> checks containment.</p>'+source("Qualitative simulation illustration.<br>The Solver uses both to select an action.")+'</div>',
loop:()=>title("Together, they form a planning loop","Four abstractions wrap the base policy without retraining.")+'<div class="pipeline-layout"><div class="pipeline-image">'+fig("ManiDreams_pipeline","Original sample-predict-constrain planning and execution pipeline")+'<svg class="pipeline-overlay" viewBox="0 0 720 259.866" preserveAspectRatio="xMidYMid meet" aria-hidden="true"><rect class="pipeline-focus" x="12" y="136" width="96" height="86" rx="10"/></svg></div><div class="loop-steps"><div class="loop-step"><strong>1 &nbsp; Sample</strong><span>Solver proposes '+math("N")+' actions</span></div><div class="loop-step"><strong>2 &nbsp; Predict</strong><span>TSIP propagates each DRIS</span></div><div class="loop-step"><strong>3 &nbsp; Constrain</strong><span>Cage scores and validates</span></div><div class="loop-step"><strong>4 &nbsp; Execute</strong><span>Act, observe, refresh DRIS</span></div></div></div>',
backends:()=>title("Two physics backends, the same task","Circular object pushing; each pair shows plan (left) and execution (right).")+'<div class="two-videos"><figure>'+video("push-sim","Object pushing with simulation TSIP")+'<figcaption class="media-title">Simulation-based TSIP</figcaption><p class="media-description">Parallel physical instances</p></figure><figure>'+video("push-learned","Object pushing with diffusion TSIP")+'<figcaption class="media-title">Diffusion-based TSIP</figcaption><p class="media-description">Prediction over a DRIS representation</p></figure></div><div class="same-stack"><strong>Unchanged</strong><span>Cage</span><span>Discrete sampler</span><span>N-best solver</span></div>',
tasks:()=>title("The same interfaces support different tasks","Changing the sampler, cage and optimizer.")+'<div class="two-videos"><figure>'+video("catch","Ball catching with a pretrained PPO sampler")+'<figcaption class="media-title">Ball catching</figcaption><p class="media-description">Pretrained PPO sampler<br>Simulation TSIP</p></figure><figure>'+video("card","Card picking with a geometric 3D cage and MPPI")+'<figcaption class="media-title">Card picking</figcaption><p class="media-description">Geometric 3D cage<br>Continuous sampling + MPPI</p></figure></div><p class="task-bottom">One planning interface, independently replaceable components.</p>',
robot:()=>title("The loop reaches the physical robot","Franka Panda · diffusion TSIP · SAM2 pixel DRIS")+'<div class="robot-videos"><div class="robot-task"><h3>Push for clearance, then pick</h3>'+robotExample("real-pick","Croissant","Croissant: real robot pushes the target out of clutter, then picks it")+robotExample("real-chicken","Chicken<br>leg","Chicken leg: a second cluttered push-picking trial")+'</div><div class="robot-task"><h3>Use a wall corner to scoop</h3>'+robotExample("real-scoop","Steak","Steak: real robot moves the flat target toward a wall corner and scoops it")+robotExample("real-box","Thin box","Thin box: a second wall-corner scooping trial")+'</div></div><div class="robot-notes"><p class="fragment" data-step="1"><strong class="blue">8-action chunks.</strong> Observe and refresh between chunks.<br>Zero-shot executor swap, without fine-tuning.</p>'+source("Camera view from beneath a transparent table.<br>Qualitative demonstrations; original x2 playback.")+'</div>',
benchmark:()=>title("The same PPO policy under perturbation",'<span style="color:#2f79b8">PPO: direct execution</span> &nbsp;&nbsp; <span style="color:#c3484c">ManiDreams: PPO as the sampler</span>')+'<div class="benchmark-figure">'+fig("Exp_1","Original nine benchmark plots: PushCube, PickCube and PushT under observation noise, delay and physics perturbations")+'<div class="benchmark-focus" aria-hidden="true"></div><div class="benchmark-zoom"><div><h3>PushT</h3><p>Increasing<br>observation delay</p><small>Success rate<br>Shared 0 to 1 scale</small></div>'+fig('PushT-delay','Original PushT observation-delay plot, enlarged without changing the data')+'</div></div>'+source(benchmarkCaption(false))+'<p class="finding">Largest gains at moderate perturbations.<br><span>At extremes, the fixed DRIS can stop covering the uncertainty.</span></p>',
tradeoff:()=>title("Coverage and compute set the trade-off","PushCube ablations and measured runtime")+'<div class="trade-layout"><div class="trade-column"><h3>Success under combined perturbations</h3><table class="ablation-table" aria-label="Ablation success rates in percent"><thead><tr><th>'+math("m")+' or '+math("N")+'</th><th>1</th><th>4</th><th class="default">8</th><th>16</th><th>32</th></tr></thead><tbody><tr><td>Vary DRIS size '+math("m")+'</td><td>58</td><td>72</td><td class="default">82</td><td>85</td><td>86</td></tr><tr><td>Vary samples '+math("N")+'</td><td>52</td><td>71</td><td class="default">82</td><td>87</td><td>88</td></tr></tbody></table><div class="width-results"><div><strong>74%</strong><span>Narrow</span></div><div class="preferred"><strong>82%</strong><span>Medium</span></div><div><strong>76%</strong><span>Wide</span></div></div><p class="trade-note">Diminishing returns above 16.<br>Medium width works best in this setting.</p>'+source("Noise "+math("\\sigma = 0.04\\,\\mathrm{m}")+"; chunk length 3; physics severity 0.4.<br>100 episodes/config. Others held at "+math("m = N = 8")+", Medium.")+'</div><div class="trade-column"><h3>Wall-clock time per step</h3><div class="runtime-figure">'+fig("computing","Measured runtime: PPO 20.2 ms, simulation 49.6 to 129.2 ms, diffusion 55.2 ms")+'</div><p class="runtime-stat fragment" data-step="1">20.2 ms PPO &nbsp; / &nbsp; 55.2 ms diffusion TSIP</p>'+source("Single NVIDIA RTX 5070 Ti laptop GPU.<br>Simulation configurations: 49.6–129.2 ms/step.")+'</div></div>',
closing:()=>'<h2>Which futures should<br>the robot accept?</h2><p class="closing-core">Keep uncertainty inside the planning loop.<br>Add robustness around the policy, without retraining.</p><div class="closing-bottom"><div><p class="closing-limits">Current limits: task-specific DRIS and cage tuning,<br>and a suitable dynamics backend.</p><p class="closing-name">Gaotian Wang<small>Rice University</small></p></div><a class="project-qr" href="https://rice-robotpi-lab.github.io/ManiDreams/" target="_blank" rel="noopener"><img src="assets/figures/qr-project.svg" alt="Scan to open the ManiDreams project page"><strong>Explore ManiDreams</strong><span>Project · Demos · Slides · Code</span></a></div>',
architecture:()=>'<div class="arch-layout"><div class="arch-figure">'+fig("arch","Original three-layer software architecture")+'</div><div class="arch-content"><div class="arch-heading">'+title("A layered framework<br>for manipulation","Separate task definitions, component implementations and interfaces.")+'</div><div class="arch-copy"><div><h3>Task integration</h3><p>Supply a scene, pipeline configuration<br>and action sampler.</p></div><div><h3>Concrete implementations</h3><p>Components share ManiDreamsEnv.</p></div><div><h3>Abstract interfaces</h3><p>DRIS, TSIP, Cage, Solver and Executor</p></div></div></div></div>',
configuration:()=>title("Evaluation settings","Backup")+'<div class="settings-layout"><div><h3>Robustness benchmark</h3><p>Same PPO policy for both methods.<br>'+math("N = 8,\\; m = 8,\\; \\lambda = 0.1")+'.</p><p>Vary one perturbation at a time.<br>Hold the other two at zero.</p><p>'+math("10")+' runs '+math("\\times100")+' episodes/configuration.<br>Report mean and '+math("\\pm1")+' standard deviation.</p></div><div><h3>Ablation</h3><p>PushCube, 100 episodes/configuration.<br>Noise '+math("\\sigma = 0.04\\,\\mathrm{m}")+', chunk length 3,<br>physics severity 0.4.</p><table><tr><th>Width</th><th>Parameter multiplier</th></tr><tr><td>Narrow</td><td>'+math("[0.9,\\,1.2]\\times")+' default</td></tr><tr><td>Medium</td><td>'+math("[0.5,\\,2]\\times")+' default</td></tr><tr><td>Wide</td><td>'+math("[0.2,\\,3]\\times")+' default</td></tr></table></div></div>',
limitations:()=>title("Scope and limitations","Backup")+'<div class="scope-list"><div><h3>Tested perturbations</h3><p>Observation noise, observation delay and physical parameters.</p></div><div><h3>Manual specification</h3><p>DRIS parameters and cages need task-specific tuning.</p></div><div><h3>Model dependence</h3><p>Discontinuous dynamics may require more expressive TSIP backends.</p></div><div><h3>Real-robot evidence</h3><p>Qualitative deployment demonstrations, without reported success rates.</p></div></div>'
};
let index=0,step=0,elapsed=0,running=false,auto=false,mediaPaused=false,mediaSerial=0,lastTick=performance.now(),presenterWindow=null,controlTimeout;
const chapters={opening:"The question",title:"ManiDreams",uncertainty:"The challenge",dris:"Represent",predict:"Predict",cage:"Constrain",teaser:"The core idea",architecture:"The framework",loop:"The planning loop",tasks:"Compose",backends:"Compose",robot:"Deploy",benchmark:"Evaluate",tradeoff:"Trade-offs",closing:"Open source"};
const elapsedAtSlide=i=>slides.slice(0,i).reduce((n,s)=>n+s.duration,0);
const format=t=>{t=Math.max(0,Math.floor(t));return String(Math.floor(t/60)).padStart(2,"0")+":"+String(t%60).padStart(2,"0")};
const timerView=t=>({over:t>TOTAL,time:format(t>TOTAL?t-TOTAL:Math.ceil(TOTAL-t))});
function updateTimer(){
  if(isPresenter||isPreview)return;
  const view=timerView(elapsed),button=$("#talk-timer-toggle");
  $("#talk-timer").classList.toggle("over",view.over);
  $("#talk-timer").dataset.state=running?"running":elapsed>0?"paused":"ready";
  $("#talk-timer-value").textContent=(view.over?"+":"")+view.time;
  $("#talk-timer-status").textContent=view.over?"Over time":running?"Left":elapsed>0?"Paused":"Start";
  $("#talk-timer-icon").textContent=running?"Ⅱ":"▶";
  button.setAttribute("aria-pressed",String(running));
  button.setAttribute("aria-label",(running?"Pause":"Start")+" timer. "+view.time+(view.over?" overtime":" remaining"));
  button.title=(running?"Pause":"Start")+" timer (T)";
}
function advanceClock(){
  const now=performance.now();
  if(running)elapsed+=Math.max(0,now-lastTick)/1000;
  lastTick=now;
}
function build(){
  $("#slides").innerHTML=slides.map((s,i)=>'<section id="'+s.id+'" class="slide '+s.id+(s.id==="title"?" title-slide":"")+(s.id==="tasks"?" task":"")+(["dris","predict","cage"].includes(s.id)?" comparison":"")+'" data-index="'+i+'" aria-label="'+esc(s.title)+'" aria-hidden="true" inert>'+templates[s.id]()+'</section>').join("");
  $("#help .help-intro").textContent=MAIN+" slides · 9 minutes · "+(slides.length-MAIN)+" backup slides";
  $("#overview-list").innerHTML=slides.map((s,i)=>(i===MAIN?'<h3 class="backup-heading">Backup slides · outside the 9-minute talk</h3>':"")+'<button class="overview-item" data-goto="'+i+'"><small>'+(i<MAIN?format(elapsedAtSlide(i))+" – "+format(elapsedAtSlide(i)+s.duration):"Backup "+(i-MAIN+1))+'</small><strong>'+String(i+1).padStart(2,"0")+' &nbsp; '+esc(s.title.replace("Backup: ",""))+'</strong></button>').join("");
  $$(".motion-media").forEach(img=>img.addEventListener("error",()=>{
    if(!img.dataset.playing)return;
    const v=document.createElement("video");
    v.className="video";v.muted=true;v.defaultMuted=true;v.autoplay=true;v.loop=true;v.playsInline=true;
    v.setAttribute("muted","");v.setAttribute("playsinline","");
    v.poster=img.dataset.poster;v.src=img.dataset.video;v.setAttribute("aria-label",img.alt);
    img.replaceWith(v);updateMedia(false);
  },{once:true}));
}
function resize(){const scale=Math.min(innerWidth/1600,innerHeight/900);$("#stage").style.transform="translate(-50%, -50%) scale("+scale+")"}
function play(v){if(isPreview)return;const promise=v.play();if(promise)promise.catch(()=>{const wrap=v.parentElement;if(!wrap.querySelector(".media-fallback")){const b=document.createElement("button");b.className="media-fallback";b.textContent="Play clip";b.onclick=()=>{v.play().then(()=>b.remove()).catch(()=>{});};wrap.append(b);}})}
function updateMedia(changed){
  $$(".slide").forEach((el,i)=>{
    const active=i===index&&!isPreview&&!isPresenter&&!document.hidden&&$("#blackout").hidden&&!mediaPaused;
    el.querySelectorAll(".motion-media").forEach(img=>{
      if(active){
        if(changed||!img.dataset.playing){img.dataset.playing="true";img.src=img.dataset.gif+"?play="+(++mediaSerial)}
      }else if(img.dataset.playing){delete img.dataset.playing;img.src=img.dataset.poster}
    });
    el.querySelectorAll("video").forEach(v=>{
      if(active){if(changed)v.currentTime=0;play(v)}else v.pause();
    });
  });
  const button=$("#media-toggle");
  button.textContent=mediaPaused?"Play clips":"Pause clips";
  button.setAttribute("aria-pressed",String(mediaPaused));
}
const focusRects=[
  // Native coordinates in the original 720 × 259.866 pipeline figure.
  // Sampler; TSIP + candidate DRIS; Cage + Optimizer; Executor.
  [12,136,96,86],[125,137,191,86],[339,130,198,91],[551,29,165,223]
];
function setPhase(){
  const current=$$(".slide")[index];current.dataset.phase=String(step);
  current.querySelectorAll(".fragment").forEach(el=>{const visible=Number(el.dataset.step)<=step;el.classList.toggle("shown",visible);el.setAttribute("aria-hidden",String(!visible))});
  if(slides[index].id==="loop"){
    const rect=focusRects[step];const box=current.querySelector(".pipeline-focus");
    ["x","y","width","height"].forEach((attr,i)=>box.style.setProperty(attr,rect[i]+"px"));
    current.querySelectorAll(".loop-step").forEach((el,i)=>el.classList.toggle("current",i===step));
  }
  if(slides[index].id==="benchmark"){
    current.querySelector('.evidence-caption').innerHTML=benchmarkCaption(step>0);
  }
}
function render(changed=false){
  $$(".slide").forEach((el,i)=>{el.classList.toggle("active",i===index);el.setAttribute("aria-hidden",String(i!==index));el.inert=i!==index;});
  const dark=slides[index].id==="closing";document.body.classList.toggle("dark",dark);
  setPhase();updateMedia(changed);
  const number=index<MAIN?String(index+1).padStart(2,"0")+" / "+MAIN:"Backup "+(index-MAIN+1);
  $("#slide-count").textContent=number;$("#nav-count").textContent=number;
  $("#chapter-label").textContent=chapters[slides[index].id]||"Backup";
  $("#progress>div").style.width=(index<MAIN?(index/(MAIN-1))*100:100)+"%";
  $$(".overview-item").forEach((el,i)=>el.setAttribute("aria-current",String(i===index)));
  $("#announcer").textContent=slides[index].title;
  if(!isPreview&&!isPresenter)history.replaceState(null,"","#"+slides[index].id+(step?"/"+step:""));
  if(!$("#rehearsal-notes").hidden)$("#rehearsal-notes").innerHTML=notesHTML();
  if(!isPreview)sendState();
}
function go(i,s=0){
  const target=Math.max(0,Math.min(slides.length-1,i)),changed=target!==index;
  index=target;step=Math.max(0,Math.min(slides[index].steps,s));
  render(changed);
}
function next(){if(step<slides[index].steps)go(index,step+1);else if(index<MAIN-1||index>=MAIN&&index<slides.length-1)go(index+1)}
function prev(){if(step>0)go(index,step-1);else if(index>0)go(index-1,slides[index-1].steps)}
function state(){return{kind:"manidreams-state",index,step,elapsed,running,auto,mediaPaused}}
function sendState(){updateTimer();if(presenterWindow&&!presenterWindow.closed)presenterWindow.postMessage(state(),"*")}
function notesHTML(i=index,s=step){
  const d=slides[i];return'<h2>'+esc(d.title)+'</h2>'+
   (d.stepLabels.length?'<div class="build-label">Build '+(s+1)+' / '+(d.steps+1)+' · '+esc(d.stepLabels[s]||"")+'</div>':"")+
   '<p class="cn-cue" lang="zh-CN">'+esc(d.cn)+'</p><h3>Talk track</h3>'+
   d.script.split("\n\n").map(t=>'<p>'+esc(t)+'</p>').join("")+
   (d.bridge?'<h3>Transition</h3><p class="bridge">'+esc(d.bridge)+'</p>':"")+
   '<details><summary>Sources & scope</summary>'+d.sources.map(t=>'<p>'+esc(t)+'</p>').join("")+'</details>';
}
function toggleTimer(){advanceClock();running=!running;sendState()}
function act(action){
  if(isPresenter){if(window.opener&&!window.opener.closed)window.opener.postMessage({kind:"manidreams-command",action},"*");return}
  if(["next","prev","home","end","overview"].includes(action))auto=false;
  switch(action){
    case"next":next();break;case"prev":prev();break;
    case"overview":$("#overview").showModal();break;case"help":$("#help").showModal();break;
    case"close":$$("dialog[open]").forEach(d=>d.close());break;
    case"fullscreen":if(document.fullscreenElement)document.exitFullscreen();else document.documentElement.requestFullscreen?.().catch(()=>{});break;
    case"presenter":{
      const u=new URL(location.href);u.search="?presenter=1";u.hash="";
      presenterWindow=window.open(u.href,"ManiDreamsPresenter","width=1440,height=940");sendState();break;
    }
    case"notes":$("#rehearsal-notes").hidden=!$("#rehearsal-notes").hidden;if(!$("#rehearsal-notes").hidden)$("#rehearsal-notes").innerHTML=notesHTML();break;
    case"timer":toggleTimer();break;
    case"resetTimer":elapsed=0;running=false;auto=false;lastTick=performance.now();sendState();break;
    case"auto":elapsed=0;auto=true;running=true;lastTick=performance.now();go(0);break;
    case"stopAuto":auto=false;sendState();break;
    case"replay":mediaPaused=false;updateMedia(true);break;
    case"media":mediaPaused=!mediaPaused;updateMedia(false);sendState();break;
    case"blackout":$("#blackout").hidden=!$("#blackout").hidden;updateMedia(false);break;
    case"home":go(0);break;case"end":go(MAIN-1);break;
  }
}
function showControls(){document.body.classList.add("controls-visible");clearTimeout(controlTimeout);controlTimeout=setTimeout(()=>document.body.classList.remove("controls-visible"),2200)}
let lastPreview="",presenterState=null;
function updatePresenter(s){
  presenterState=s;index=s.index;step=s.step;elapsed=s.elapsed;running=s.running;auto=s.auto;
  $("#connection").textContent=s.auto?"Timed rehearsal running":"Connected to audience view";
  const clock=timerView(s.elapsed);
  $("#elapsed").textContent=format(s.elapsed);$("#remaining").textContent=clock.time+(clock.over?" over time":" remaining");
  $(".clock-group").classList.toggle("over",clock.over);
  const marker=index+":"+step;
  if(marker!==lastPreview){
    $("#speaker-notes").innerHTML=notesHTML();$("#speaker-notes").scrollTop=0;
    $("#speaker-position").textContent=(index<MAIN?"Slide "+(index+1)+" / "+MAIN:"Backup "+(index-MAIN+1))+" · build "+(step+1);
    const start=elapsedAtSlide(index),end=start+slides[index].duration;
    $("#schedule").textContent=index<MAIN?"Target: "+format(start)+"–"+format(end)+" · "+slides[index].duration+" seconds on this slide":"Backup slide · outside the nine-minute talk";
    sendPreviews();lastPreview=marker;
  }
}
function sendPreviews(){
  if(!presenterState)return;
  $("#current-preview").contentWindow?.postMessage({kind:"manidreams-preview",index,step},"*");
  $("#next-preview").contentWindow?.postMessage({kind:"manidreams-preview",index:Math.min(index+1,index<MAIN?MAIN-1:slides.length-1),step:0},"*");
}
build();resize();addEventListener("resize",resize);
if(isPresenter){
  document.body.classList.add("presenter-mode");$("#presenter").hidden=false;
  const u=new URL(location.href);u.search="?preview=1";u.hash="";
  $("#current-preview").src=u.href;$("#next-preview").src=u.href;
  $("#current-preview").addEventListener("load",sendPreviews);$("#next-preview").addEventListener("load",sendPreviews);
  window.opener?.postMessage({kind:"manidreams-ready"},"*");
  setTimeout(()=>{if(!presenterState)$("#connection").textContent="Open Presenter from the audience window (P)."},3000);
}else{
  if(isPreview)document.body.classList.add("preview-mode");
  const [id,phase]=location.hash.slice(1).split("/");go(Math.max(0,slides.findIndex(s=>s.id===id)),Number(phase)||0);
}
addEventListener("message",e=>{
  const d=e.data;if(!d||typeof d.kind!=="string")return;
  if(isPreview&&d.kind==="manidreams-preview"&&e.source===parent)go(d.index,d.step);
  if(isPresenter&&d.kind==="manidreams-state"&&e.source===window.opener)updatePresenter(d);
  if(!isPresenter&&!isPreview){
    if(d.kind==="manidreams-ready"&&e.source===presenterWindow)sendState();
    if(d.kind==="manidreams-command"&&e.source===presenterWindow)act(d.action);
  }
});
document.addEventListener("click",e=>{
  const action=e.target.closest("[data-action]");if(action){act(action.dataset.action);if(e.detail>0&&action.closest("#talk-timer"))action.blur();return}
  const command=e.target.closest("[data-command]");if(command){act(command.dataset.command);return}
  const target=e.target.closest("[data-goto]");if(target){auto=false;go(Number(target.dataset.goto));$("#overview").close();return}
});
document.addEventListener("keydown",e=>{
  if(e.target.matches("input,textarea,select")||e.ctrlKey||e.metaKey||e.altKey)return;
  if($("dialog[open]"))return;
  const k=e.key.toLowerCase();const map={arrowright:"next"," ":"next",pagedown:"next",arrowleft:"prev",pageup:"prev",f:"fullscreen",p:"presenter",o:"overview","?":"help",h:"help",n:"notes",t:"timer",r:"replay",m:"media",b:"blackout",home:"home",end:"end"};
  if(e.target.closest("#talk-timer")&&(k===" "||k==="enter"))return;
  if(k==="escape"){if(!$("#blackout").hidden)act("blackout");$("#rehearsal-notes").hidden=true;return}
  if(map[k]){e.preventDefault();act(map[k]);if(!isPresenter)showControls()}
});
document.addEventListener("mousemove",showControls,{passive:true});
let touchX=0,touchY=0;
$("#viewport").addEventListener("touchstart",e=>{touchX=e.changedTouches[0].clientX;touchY=e.changedTouches[0].clientY},{passive:true});
$("#viewport").addEventListener("touchend",e=>{const dx=e.changedTouches[0].clientX-touchX,dy=e.changedTouches[0].clientY-touchY;if(Math.abs(dx)>65&&Math.abs(dy)<70)act(dx<0?"next":"prev")},{passive:true});
document.addEventListener("visibilitychange",()=>{if(!isPresenter)updateMedia(false)});
setInterval(()=>{
  if(isPresenter||isPreview)return;
  advanceClock();
  if(auto&&running){
    if(elapsed>=TOTAL){go(MAIN-1);auto=false;running=false;elapsed=TOTAL}
    else{let acc=0,chosen=0;for(let i=0;i<MAIN;i++){if(elapsed<acc+slides[i].duration){chosen=i;break}acc+=slides[i].duration}
      const phase=Math.min(slides[chosen].steps,Math.floor((elapsed-acc)/slides[chosen].duration*(slides[chosen].steps+1)));
      if(chosen!==index||phase!==step)go(chosen,phase);}
  }
  sendState();
},250);
showControls();
// Expose a small read-only state plus navigation for browser QA and rehearsal links.
window.ManiDreams={go,state,slides,mainCount:MAIN,totalSeconds:TOTAL,action:act};
})();
