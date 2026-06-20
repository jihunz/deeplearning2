"""발표 그림 재제작 (tiny-budget-ada 디자인 문법: Pretendard, Transformer=blue/LSTM=회색, 직접 라벨 + 인사이트)."""
import warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
for w in ['Regular','SemiBold','Bold']:
    font_manager.fontManager.addfont(f'/Users/jihunjang/Library/Fonts/Pretendard-{w}.ttf')
plt.rcParams['font.family']='Pretendard'; plt.rcParams['axes.unicode_minus']=False

BLUE="#1D4ED8"; GRAY="#6B7280"; INK="#111827"; SUB="#94A3B8"; RED="#C0392B"; LINE="#E5E7EB"
R="/Users/jihunjang/workspace/ust/deeplearning2/term-project-lstm-vs-transformer-agnews/redesign"
def clean(ax):
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    ax.spines['left'].set_color(LINE); ax.spines['bottom'].set_color(LINE)
    ax.tick_params(colors=SUB, labelsize=10)

# ── fig_loss: train/val loss (LSTM 회색, Transformer blue; LSTM val 폭증 red)
ep=list(range(8))
lstm_tr=[0.40,0.22,0.15,0.10,0.07,0.055,0.04,0.035]; lstm_val=[0.54,0.55,0.65,0.95,1.55,1.68,1.99,1.73]
tr_tr=[0.44,0.27,0.217,0.18,0.154,0.134,0.117,0.103]; tr_val=[0.34,0.288,0.292,0.285,0.288,0.312,0.32,0.33]
fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.2))
a1.plot(ep,lstm_tr,color=GRAY,marker='o',ms=5,lw=2); a1.plot(ep,lstm_val,color=RED,marker='s',ms=5,lw=2)
a1.text(7.05,0.035,'train',color=GRAY,fontsize=10,va='center',fontweight='bold')
a1.text(6.5,1.73,'validation',color=RED,fontsize=10,va='center',ha='right',fontweight='bold')
a1.annotate("val loss 폭증 = overfitting",xy=(6,1.99),xytext=(2.3,1.78),fontsize=10.5,color=RED,
    arrowprops=dict(arrowstyle='->',color=RED,lw=1.3))
a1.set_title("LSTM",fontsize=13.5,color=GRAY,fontweight='bold',loc='left',pad=10)
a2.plot(ep,tr_tr,color=BLUE,marker='o',ms=5,lw=2); a2.plot(ep,tr_val,color=BLUE,marker='s',ms=5,lw=2,ls=(0,(4,3)),alpha=0.55)
a2.text(7.05,0.103,'train',color=BLUE,fontsize=10,va='center',fontweight='bold')
a2.text(7.05,0.33,'validation',color=BLUE,fontsize=10,va='center',alpha=0.8,fontweight='bold')
a2.annotate("val 안정 = 일반화",xy=(7,0.33),xytext=(3.2,0.40),fontsize=10.5,color=BLUE,
    arrowprops=dict(arrowstyle='->',color=BLUE,lw=1.3))
a2.set_title("Transformer",fontsize=13.5,color=BLUE,fontweight='bold',loc='left',pad=10)
for ax in (a1,a2): clean(ax); ax.set_xlabel("epoch",fontsize=11,color=INK); ax.grid(axis='y',color='#F1F5F9',lw=1); ax.set_axisbelow(True); ax.set_xlim(-0.3,8.2)
a1.set_ylabel("loss",fontsize=11,color=INK)
fig.tight_layout(pad=1.2); fig.savefig(R+"/fig_loss.png",dpi=200,bbox_inches='tight',facecolor='white'); print("fig_loss")

# ── fig_confusion: 4x4 heatmap
labels=['World','Sports','Business','Sci/Tech']
lstm_cm=np.array([[1567,40,160,133],[49,1517,171,163],[62,8,1572,258],[60,11,154,1675]])
tr_cm=np.array([[1723,53,74,50],[22,1862,11,5],[52,32,1669,147],[74,29,138,1659]])
fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.7))
for ax,cm,title,c in [(a1,lstm_cm,'LSTM   acc 0.833',GRAY),(a2,tr_cm,'Transformer   acc 0.910',BLUE)]:
    ax.imshow(cm,cmap='Blues',vmin=0,vmax=1900)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(labels,rotation=25,ha='right',fontsize=10); ax.set_yticklabels(labels,fontsize=10)
    for i in range(4):
        for j in range(4):
            v=cm[i,j]
            ax.text(j,i,f"{v:,}",ha='center',va='center',fontsize=10,fontweight=('bold' if i==j else 'normal'),
                color=('white' if v>950 else INK))
    ax.set_title(title,fontsize=12.5,color=c,fontweight='bold',loc='left',pad=12)
    ax.set_xlabel("predicted",fontsize=10.5,color=INK)
    for sp in ax.spines.values(): sp.set_color(LINE)
    ax.tick_params(length=0)
a1.set_ylabel("true",fontsize=10.5,color=INK)
fig.tight_layout(pad=1.2); fig.savefig(R+"/fig_confusion.png",dpi=200,bbox_inches='tight',facecolor='white'); print("fig_confusion")

# ── fig_ablation: data efficiency
pct=[5,25,50,100]; lstm_a=[0.727,0.846,0.833,0.833]; tr_a=[0.798,0.875,0.897,0.910]
fig,ax=plt.subplots(figsize=(8,4.7))
ax.plot(pct,lstm_a,color=GRAY,marker='o',ms=9,lw=2.4); ax.plot(pct,tr_a,color=BLUE,marker='s',ms=9,lw=2.4)
ax.text(101,0.833,' LSTM',color=GRAY,fontsize=12,va='center',fontweight='bold')
ax.text(101,0.910,' Transformer',color=BLUE,fontsize=12,va='center',fontweight='bold')
for x,y in zip(pct,lstm_a): ax.text(x,y-0.011,f"{y:.3f}",color=GRAY,fontsize=9.5,ha='center',va='top')
for x,y in zip(pct,tr_a): ax.text(x,y+0.009,f"{y:.3f}",color=BLUE,fontsize=9.5,ha='center',va='bottom',fontweight='bold')
ax.annotate("25%에서 포화",xy=(25,0.846),xytext=(33,0.792),fontsize=11,color=GRAY,
    arrowprops=dict(arrowstyle='->',color=GRAY,lw=1.3))
clean(ax); ax.set_xlabel("training data (%)",fontsize=11.5,color=INK); ax.set_ylabel("test accuracy",fontsize=11.5,color=INK)
ax.set_xticks(pct); ax.grid(axis='y',color='#F1F5F9',lw=1); ax.set_axisbelow(True); ax.set_xlim(0,118); ax.set_ylim(0.70,0.93)
fig.tight_layout(pad=1.2); fig.savefig(R+"/fig_ablation.png",dpi=200,bbox_inches='tight',facecolor='white'); print("fig_ablation")

# ── fig_mechanism: 토큰 중요도 (Sports 예시, LSTM 오답 vs Transformer 정답)
lt=['giddy','phelps','touches','gold','time','for','first','michael']; lv=[0.225,0.185,0.123,0.115,0.062,0.051,0.051,0.038]
st=['phelps','seconds','touches','giddy','time','medley','400','won']; sv=[0.122,0.087,0.084,0.066,0.058,0.057,0.056,0.053]
at=['400','giddy','gold','seconds','time','phelps','medley','minutes']; av=[0.165,0.13,0.124,0.084,0.062,0.05,0.047,0.046]
fig,axs=plt.subplots(1,3,figsize=(13,4.0))
panels=[(axs[0],lt,lv,"LSTM gradient saliency","예측: Sci/Tech (오답)",GRAY,'giddy'),
        (axs[1],st,sv,"Transformer gradient saliency","예측: Sports (정답)",BLUE,None),
        (axs[2],at,av,"Transformer attention weight","예측: Sports (정답)",BLUE,None)]
for ax,toks,vals,title,sub,c,hl in panels:
    y=np.arange(len(toks))[::-1]
    cols=[RED if (hl and t==hl) else c for t in toks]
    ax.barh(y,vals,color=cols,height=0.7)
    ax.set_yticks(y); ax.set_yticklabels(toks,fontsize=10.5)
    tc=RED if '오답' in sub else c
    ax.set_title(f"{title}\n{sub}",fontsize=11,color=tc,fontweight='bold',loc='left',pad=10,linespacing=1.5)
    clean(ax); ax.tick_params(axis='x',labelsize=9)
axs[0].annotate("주제와 무관한 'giddy'에 쏠림",xy=(0.225,7),xytext=(0.07,5.3),fontsize=10,color=RED,
    arrowprops=dict(arrowstyle='->',color=RED,lw=1.2))
fig.suptitle("같은 Sports 기사: LSTM은 한 토큰에 쏠려 오답, Transformer는 주제 토큰에 분산",
    fontsize=12.5,color=INK,fontweight='bold',y=1.14,x=0.5)
fig.tight_layout(pad=1.0); fig.savefig(R+"/fig_mechanism.png",dpi=200,bbox_inches='tight',facecolor='white'); print("fig_mechanism")
