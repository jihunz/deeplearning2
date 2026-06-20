/**
 * LSTM vs Transformer 발표자료 v2 — tiny-budget-ada 디자인 문법 (PowerPoint 기본 불릿 본문)
 * 본문 문법: body() — 한 텍스트 박스에 [헤더(blue) + 불릿 항목(ppt 기본 bullet + indentLevel)].
 *   PowerPoint에서 Tab 들여쓰기·자동 불릿 그대로 동작. 강조는 항목 단위(핵심 항목 전체를 진하게).
 *   (pptxgenjs는 한 paragraph에서 부분강조+불릿을 동시 지원하지 못해 항목은 단일 run으로 둔다.)
 * 실행: NODE_PATH=$(npm root -g) node redesign/build.js
 */
const pptxgen=require("pptxgenjs");
const p=new pptxgen();
p.defineLayout({name:"W",width:13.333,height:7.5});p.layout="W";p.author="장지훈";
const F="Pretendard",FSB="Pretendard SemiBold";
const C={ink:"111827",body:"374151",sub:"6B7280",blue:"1D4ED8",green:"0F766E",
  line:"E5E7EB",bg:"FFFFFF",off:"F8FAFC",bpale:"EFF4FF"};
const MX=0.5,W=13.333-MX*2;
const FIG="/Users/jihunjang/workspace/ust/deeplearning2/term-project-lstm-vs-transformer-agnews/redesign";
const OUT="/Users/jihunjang/workspace/ust/deeplearning2/term-project-lstm-vs-transformer-agnews/deliverables/5_presentation_v2.pptx";
const STEPS=["배경","방법","결과","결론"], TOT=10;

function head(s,active,runs){
  s.background={color:C.bg};
  const parts=[];
  STEPS.forEach((t,i)=>{parts.push({text:t,options:{color:i===active?C.blue:"BFC6D0",bold:i===active}});
    if(i<STEPS.length-1)parts.push({text:"     ",options:{color:"BFC6D0"}});});
  s.addText(parts,{x:MX,y:0.36,w:9,h:0.3,fontFace:F,fontSize:12.5,margin:0});
  s.addText(runs,{x:MX,y:0.74,w:W,h:0.5,fontFace:F,fontSize:21,bold:true,color:C.ink,margin:0,lineSpacingMultiple:1.0});
}
function pg(s,n){ s.addText(n+" / "+TOT,{x:11.9,y:7.12,w:1.1,h:0.26,fontFace:F,fontSize:9.5,color:"AAB2BD",align:"right",margin:0}); }
// 본문: blocks=[{h:"헤더", items:[ "평문" | {t,b,c,lv} ]}]. 항목=단일 run + ppt 기본 불릿(indentLevel).
function body(s,x,y,w,h,blocks,fs,gap){
  fs=fs||14; gap=gap||14;
  const runs=[];
  blocks.forEach((b,bi)=>{
    if(b.h) runs.push({text:b.h,options:{bold:true,color:C.blue,fontSize:fs+1.5,breakLine:true,paraSpaceBefore:bi>0?gap:0,paraSpaceAfter:7}});
    b.items.forEach(it=>{
      let t,lv=0,extra={};
      if(typeof it==="string"){ t=it; }
      else { t=it.t; lv=it.lv||0; if(it.b)extra.bold=true; if(it.c)extra.color=it.c; }
      runs.push({text:t,options:Object.assign({color:C.body,fontSize:fs,breakLine:true,paraSpaceAfter:5,
        lineSpacingMultiple:1.25,bullet:{code:lv?"2013":"25CF",indent:18},indentLevel:lv+1},extra)});
    });
  });
  s.addText(runs,{x,y,w,h,fontFace:F,margin:0,valign:"top"});
}

// ── 1 표지 (메타 프레임)
let s=p.addSlide(); s.background={color:C.bg};
const MXC=0.9,WC=13.333-MXC*2;
s.addShape(p.shapes.RECTANGLE,{x:0,y:0,w:0.16,h:7.5,fill:{color:C.blue}});
s.addText("DEEP LEARNING · TERM PROJECT",{x:MXC,y:0.68,w:8,h:0.35,fontFace:F,fontSize:14,bold:true,color:C.blue,charSpacing:1,margin:0});
s.addText("2026",{x:MXC+WC-3,y:0.68,w:3,h:0.35,fontFace:F,fontSize:14,color:C.sub,align:"right",margin:0});
s.addShape(p.shapes.LINE,{x:MXC,y:1.2,w:WC,h:0,line:{color:C.line,width:1}});
s.addText([
  {text:"LSTM ",options:{color:C.ink}},{text:"vs",options:{color:C.blue}},{text:" Transformer",options:{color:C.ink}},
],{x:MXC,y:2.05,w:WC,h:1.0,fontFace:F,fontSize:50,bold:true,margin:0});
s.addText("Encoder, 텍스트 분류에서의 통제된 비교",{x:MXC,y:3.18,w:WC,h:0.7,fontFace:F,fontSize:31,bold:true,color:C.ink,margin:0});
s.addText("AG News 4-class 주제 분류, 두 시퀀스 인코더를 동일 조건에서 비교",{x:MXC,y:4.22,w:WC,h:0.5,fontFace:F,fontSize:19,color:C.sub,margin:0});
s.addShape(p.shapes.LINE,{x:MXC,y:5.42,w:WC,h:0,line:{color:C.line,width:1}});
const LWC=1.55,vyc=5.72;
s.addText("발표자",{x:MXC,y:vyc+0.02,w:LWC,h:0.35,fontFace:F,fontSize:15,bold:true,color:C.blue,margin:0,valign:"top"});
s.addText([{text:"장지훈",options:{fontFace:F,bold:true,fontSize:17,color:C.ink}},{text:"   (ETRI School · 02521122)",options:{fontFace:F,fontSize:13,color:C.sub}}],{x:MXC+LWC,y:vyc,w:WC-LWC,h:0.4,margin:0,valign:"top"});
s.addText("소속",{x:MXC,y:vyc+0.64,w:LWC,h:0.35,fontFace:F,fontSize:15,bold:true,color:C.blue,margin:0,valign:"top"});
s.addText("과학기술연합대학원대학교 ETRI 스쿨",{x:MXC+LWC,y:vyc+0.62,w:WC-LWC,h:0.35,fontFace:FSB,fontSize:16,color:C.body,margin:0,valign:"top"});

// ── 2 INTRODUCTION (배경)
s=p.addSlide();
head(s,0,[{text:"정확도 경쟁이 아니라, ",options:{}},{text:"두 구조가 왜 다른지",options:{color:C.blue}},{text:"를 규명",options:{}}]);
body(s,MX,1.55,W,5.2,[
  {h:"과제",items:[
    {t:"AG News 4-class 뉴스 주제 분류 (World·Sports·Business·Sci-Tech)",b:true,c:C.ink},
    "LSTM과 Transformer Encoder를 모두 from scratch로 학습",
  ]},
  {h:"방법",items:[
    {t:"데이터·전처리·어휘·시퀀스 길이·학습 예산을 동일하게 고정",b:true,c:C.ink},
    "encoder 구조만 유일한 변수로 둔다",
  ]},
  {h:"질문",items:[
    "성능·수렴·data efficiency·오분류 양상이 어떻게 다른가",
    "그리고 그 차이의 원인은 무엇인가",
  ]},
],15,15);
pg(s,2);

// ── 3 DATASET (배경)
s=p.addSlide();
head(s,0,[{text:"balanced 4-class, 어휘는 ",options:{}},{text:"train에서만 구축",options:{color:C.blue}},{text:" (leakage 방지)",options:{}}]);
body(s,MX,1.55,W,5.2,[
  {h:"분할 (seed 42)",items:[
    {t:"Train 108,000 / Validation 12,000 / Test 7,600",b:true,c:C.ink},
  ]},
  {h:"클래스",items:[
    {t:"World, Sports, Business, Sci-Tech 4개, balanced",b:true,c:C.ink},
    "train 약 27,000 / test 1,900 each",
  ]},
  {h:"전처리 (두 모델 공통)",items:[
    "단어 단위 tokenization (소문자화)",
    "vocabulary: train-only, 상한 20,000, min frequency 2",
    "max length 128, padding 위치는 masking으로 무시",
  ]},
],15,14);
pg(s,3);

// ── 4 MODELS (방법)
s=p.addSlide();
head(s,1,[{text:"공유 골격, ",options:{}},{text:"encoder만 교체",options:{color:C.blue}}]);
body(s,MX,1.55,W,5.2,[
  {h:"공유 골격",items:[
    {t:"Embedding(128) → Encoder → masked mean pooling → Linear(4)",b:true,c:C.ink},
    "encoder 한 조각만 교체, 나머지는 모두 동일",
  ]},
  {h:"encoder 비교",items:[
    {t:"LSTM : bidirectional · 2 layers · hidden 128 / output 256 / params 3,220,484",c:C.body},
    {t:"Transformer : 2 layers · 4 heads · FFN 256 · sinusoidal PE / dim 128 / params 2,825,476",c:C.body},
  ]},
  {h:"파라미터 규모",items:[
    "임베딩이 지배적이라 두 모델 파라미터는 약 14% 차로 비슷",
  ]},
],15,15);
pg(s,4);

// ── 5 EXPERIMENTS (방법)
s=p.addSlide();
head(s,1,[{text:"유일한 변수 = ",options:{}},{text:"encoder 구조",options:{color:C.blue}},{text:" (나머지 전부 고정)",options:{}}]);
body(s,MX,1.55,W,5.2,[
  {h:"고정 (fair comparison)",items:[
    "split · tokenizer · vocabulary · max length · pooling",
    "optimizer · learning rate · batch size · epoch · seed",
  ]},
  {h:"학습",items:[
    "Adam, learning rate 0.001, batch 64, 최대 8 epochs, dropout 0.1, seed 42",
    "model selection은 validation, test는 1회",
  ]},
  {h:"Ablation · 지표",items:[
    {t:"training set 5 / 25 / 50 / 100% (vocabulary 고정, stratified, val·test 전체)",b:true,c:C.ink},
    "accuracy · macro F1-score · loss curve · confusion matrix",
  ]},
],15,14);
pg(s,5);

// ── 6 RESULTS 메인 (결과) — fig_loss 전폭
s=p.addSlide();
head(s,2,[{text:"Transformer ",options:{}},{text:"0.910",options:{color:C.blue}},{text:" vs LSTM ",options:{}},{text:"0.833",options:{color:C.sub}},{text:", 격차의 원인은 overfitting",options:{}}]);
s.addImage({path:FIG+"/fig_loss.png",x:MX,y:1.5,w:W,h:4.25,sizing:{type:"contain",w:W,h:4.25}});
body(s,MX,5.95,W,1.4,[
  {h:"",items:[
    {t:"Transformer : test accuracy 0.910 · macro F1 0.909",b:true,c:C.blue},
    {t:"LSTM : 0.833 · macro F1 0.835, val loss 폭증으로 best epoch 2 저장",c:C.body},
  ]},
],14,0);
pg(s,6);

// ── 7 RESULTS 혼동 (결과) — fig_confusion 전폭
s=p.addSlide();
head(s,2,[{text:"두 모델 공통 ",options:{}},{text:"Business ↔ Sci/Tech 혼동",options:{color:C.blue}}]);
s.addImage({path:FIG+"/fig_confusion.png",x:MX,y:1.5,w:W,h:4.0,sizing:{type:"contain",w:W,h:4.0}});
body(s,MX,5.75,W,1.5,[
  {h:"",items:[
    {t:"공통 오답 428건 : 61%가 Business/Sci-Tech 경계 (의미 중첩에 따른 모호성)",b:true,c:C.ink},
    {t:"LSTM 단독 841건 : 전 class에 분산, Transformer 단독(259건)의 3.2배",c:C.body},
  ]},
],14,0);
pg(s,7);

// ── 8 ABLATION (결과) — fig_ablation 좌 + body 우
s=p.addSlide();
head(s,2,[{text:"LSTM은 ",options:{}},{text:"25%에서 포화",options:{color:C.sub}},{text:", Transformer는 ",options:{}},{text:"계속 향상",options:{color:C.blue}}]);
s.addImage({path:FIG+"/fig_ablation.png",x:MX,y:1.7,w:7.1,h:4.8,sizing:{type:"contain",w:7.1,h:4.8}});
body(s,7.9,1.75,W-7.4,4.8,[
  {h:"Transformer",items:[
    {t:"데이터에 따라 단조 향상",b:true,c:C.blue},
    "scaling: 더 줄수록 더 좋아짐",
  ]},
  {h:"LSTM",items:[
    {t:"25%에서 정점 후 정체",b:true,c:C.ink},
    "overfitting으로 추가 데이터를 못 씀",
  ]},
  {h:"가설 기각",items:[
    "'data-hungry' 가설 기각",
    "전 구간에서 Transformer 우위",
  ]},
],14,16);
pg(s,8);

// ── 9 EXTENSION (결과) — fig_mechanism 전폭
s=p.addSlide();
head(s,2,[{text:"bag-of-words 과제, 격차는 ",options:{}},{text:"순서가 아닌 robustness",options:{color:C.blue}}]);
s.addImage({path:FIG+"/fig_mechanism.png",x:MX,y:1.45,w:W,h:3.6,sizing:{type:"contain",w:W,h:3.6}});
body(s,MX,5.25,W,1.6,[
  {h:"",items:[
    {t:"PE on/off : with-PE 0.910 vs no-PE 0.911 (차이 -0.001) → 순서 정보가 거의 불필요",c:C.body},
    {t:"robustness : LSTM은 비주제어 'giddy'에 쏠려 오답, Transformer는 주제 토큰에 분산해 정답",c:C.body},
  ]},
],14,0);
pg(s,9);

// ── 10 CONCLUSION (결론)
s=p.addSlide();
head(s,3,[{text:"동일 조건에서 ",options:{}},{text:"Transformer 우수",options:{color:C.blue}},{text:", 원인은 generalization",options:{}}]);
body(s,MX,1.55,W,5.2,[
  {h:"성능",items:[
    {t:"동일 조건에서 test accuracy 0.910 vs 0.833 (encoder 구조만 바꾼 통제된 비교)",b:true,c:C.ink},
  ]},
  {h:"원인",items:[
    {t:"generalization 차이 : LSTM은 overfitting하여 25%에서 포화, Transformer는 안정적으로 확장",b:true,c:C.ink},
  ]},
  {h:"과제 성격",items:[
    {t:"bag-of-words : positional encoding 제거해도 정확도 동일 → 격차는 순서가 아닌 robustness",b:true,c:C.ink},
  ]},
  {h:"한계 · 향후",items:[
    "단일 seed → 다중 seed 재확인, LSTM regularization 강화",
    "순서가 중요한 과제에서 positional encoding 재검증",
  ]},
],15,14);
pg(s,10);

p.writeFile({fileName:OUT}).then(()=>console.log("saved "+OUT));
