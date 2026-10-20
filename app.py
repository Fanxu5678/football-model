import ioapp.py
import math
import requests
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson

st.set_page_config(page_title='足球概率实验室', page_icon='⚽', layout='wide')
st.title('⚽ 足球概率实验室')
st.caption('独立泊松基线模型｜胜平负、体彩让球、历史滚动回测｜仅供研究，不保证命中率')

@st.cache_data(ttl=3600)
def get_data(league, season):
    url = f'https://www.football-data.co.uk/mmz4281/{season}/{league}.csv'
    r = requests.get(url, timeout=20, headers={'User-Agent':'Mozilla/5.0'})
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content), on_bad_lines='skip')
    needed = ['Date','HomeTeam','AwayTeam','FTHG','FTAG']
    if not all(c in df.columns for c in needed):
        raise ValueError('数据缺少必要字段，请切换赛季或联赛。')
    df = df[needed].copy()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    for c in ['FTHG','FTAG']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna().sort_values('Date').reset_index(drop=True)
    return df

def fit_rates(history, home, away, n=12):
    # Shrink small samples toward league-wide home and away goal averages.
    league_h = max(float(history.FTHG.mean()), .15)
    league_a = max(float(history.FTAG.mean()), .15)
    hh = history[history.HomeTeam == home].tail(n)
    aa = history[history.AwayTeam == away].tail(n)
    if len(hh) < 3 or len(aa) < 3:
        return None
    prior = 6.0
    h_att = (hh.FTHG.sum()+prior*league_h)/(len(hh)+prior)/league_h
    h_def = (hh.FTAG.sum()+prior*league_a)/(len(hh)+prior)/league_a
    a_att = (aa.FTAG.sum()+prior*league_a)/(len(aa)+prior)/league_a
    a_def = (aa.FTHG.sum()+prior*league_h)/(len(aa)+prior)/league_h
    return float(np.clip(league_h*h_att*a_def,.15,5)), float(np.clip(league_a*a_att*h_def,.15,5))

def probabilities(lh, la, handicap=-1):
    # 0..14 goals, normalized to account for negligible upper tail.
    k = np.arange(15)
    m = np.outer(poisson.pmf(k,lh), poisson.pmf(k,la))
    m /= m.sum()
    d = k[:,None]-k[None,:]
    standard = [float(m[d>0].sum()),float(m[d==0].sum()),float(m[d<0].sum())]
    adj = d + handicap
    handicap_probs = [float(m[adj>0].sum()),float(m[adj==0].sum()),float(m[adj<0].sum())]
    scores = sorted([(f'{i}-{j}',float(m[i,j])) for i in range(8) for j in range(8)],key=lambda x:-x[1])[:8]
    return standard, handicap_probs, scores

leagues={'英超':'E0','英冠':'E1','西甲':'SP1','意甲':'I1','德甲':'D1','法甲':'F1'}
league_name = st.selectbox('联赛',list(leagues))
season = st.selectbox('赛季（例如2526表示2025-26）',['2627','2526','2425','2324','2223'],index=1)
try:
    data = get_data(leagues[league_name],season)
except Exception as exc:
    st.error(f'下载数据失败：{exc}。可更换赛季重试；部分新赛季数据尚未发布。')
    st.stop()
if len(data)<40:
    st.warning('这个赛季样本太少，模型预测可能不可靠。')
st.write(f'已载入 **{len(data)}** 场已结束比赛；最近记录：{data.Date.max().date()}')
teams=sorted(set(data.HomeTeam).union(data.AwayTeam))
tab1,tab2=st.tabs(['赛前概率','历史回测'])
with tab1:
    c1,c2,c3=st.columns(3)
    home=c1.selectbox('主队',teams,index=0)
    away=c2.selectbox('客队',teams,index=min(1,len(teams)-1))
    handicap=c3.selectbox('体彩主队让球',list(range(-5,6)),index=4)
    if home==away:
        st.warning('主队和客队不能相同。')
    else:
        rates=fit_rates(data,home,away)
        if rates is None:
            st.warning('样本不足，无法给出可靠的基线概率。')
        else:
            std,hcp,scores=probabilities(*rates,handicap)
            st.subheader(f'{home} VS {away}')
            st.caption(f'预计进球：主队 {rates[0]:.2f} / 客队 {rates[1]:.2f}（历史统计估计，不是实时xG）')
            st.write('**普通胜平负**')
            a,b,c=st.columns(3)
            for col,name,p in zip([a,b,c],['胜','平','负'],std): col.metric(name,f'{p:.1%}')
            st.write(f'**体彩让球 {handicap:+d}**')
            a,b,c=st.columns(3)
            for col,name,p in zip([a,b,c],['让胜','让平','让负'],hcp): col.metric(name,f'{p:.1%}')
            st.dataframe(pd.DataFrame(scores,columns=['比分','概率']).assign(概率=lambda x:x['概率'].map(lambda p:f'{p:.1%}')),hide_index=True,use_container_width=True)
            st.info('这是基础统计模型，不包含实时首发、伤停、跨联赛实力差异；切勿把最高概率理解为必然发生。')
with tab2:
    st.write('严格按日期滚动：每场只使用它之前的比赛数据，避免未来信息泄漏。')
    min_games=st.slider('开始回测前的最低历史场次数',40,200,80,10)
    if st.button('运行回测',type='primary'):
        rows=[]
        for i in range(min_games,len(data)):
            game=data.iloc[i]
            history=data.iloc[:i]
            rates=fit_rates(history,game.HomeTeam,game.AwayTeam)
            if rates is None: continue
            std,hcp,_=probabilities(*rates,-1)
            actual=int(np.argmax([game.FTHG>game.FTAG,game.FTHG==game.FTAG,game.FTHG<game.FTAG]))
            delta=game.FTHG-game.FTAG-1
            actual_h=0 if delta>0 else (1 if delta==0 else 2)
            rows.append({'日期':game.Date,'比赛':f'{game.HomeTeam} - {game.AwayTeam}', '胜平负预测': ['胜','平','负'][int(np.argmax(std))], '胜平负实际':['胜','平','负'][actual], '胜平负命中':int(np.argmax(std)==actual),'让-1预测':['让胜','让平','让负'][int(np.argmax(hcp))], '让-1实际':['让胜','让平','让负'][actual_h], '让-1命中':int(np.argmax(hcp)==actual_h),'Brier':sum((p-int(j==actual))**2 for j,p in enumerate(std))})
        if not rows:
            st.warning('有效回测比赛不足，尝试更早的赛季。')
        else:
            result=pd.DataFrame(rows)
            a,b,c,d=st.columns(4)
            a.metric('回测场次',len(result))
            b.metric('胜平负命中率',f'{result["胜平负命中"].mean():.1%}')
            c.metric('让-1命中率',f'{result["让-1命中"].mean():.1%}')
            d.metric('平均Brier分数',f'{result.Brier.mean():.3f}')
            st.caption('Brier分数越低越好。命中率不代表投注收益；不同让球档位需要分别回测。')
            st.dataframe(result.drop(columns=['Brier']),hide_index=True,use_container_width=True)
            st.download_button('下载回测CSV',result.to_csv(index=False).encode('utf-8-sig'),'backtest.csv','text/csv')
st.caption('数据：football-data.co.uk 公开CSV。遵守数据来源使用条款。此版本只支持同一联赛内球队。')
