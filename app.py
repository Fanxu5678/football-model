import io
import math
import requests
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson

# 中文球队别名；最终仍以当前数据源中的英文队名为准。
TEAM_ALIASES = {
    '阿森纳':'Arsenal','切尔西':'Chelsea','利物浦':'Liverpool','埃弗顿':'Everton',
    '曼城':'Man City','曼联':'Man United','热刺':'Tottenham','纽卡':'Newcastle',
    '纽卡斯尔':'Newcastle','维拉':'Aston Villa','阿斯顿维拉':'Aston Villa',
    '布莱顿':'Brighton','西汉姆':'West Ham','西汉姆联':'West Ham',
    '狼队':'Wolves','富勒姆':'Fulham','水晶宫':'Crystal Palace',
    '伯恩茅斯':'Bournemouth','诺丁汉森林':"Nott'm Forest",'布伦特福德':'Brentford',
    '利兹联':'Leeds','桑德兰':'Sunderland','伯恩利':'Burnley',
    '皇马':'Real Madrid','皇家马德里':'Real Madrid','巴萨':'Barcelona',
    '巴塞罗那':'Barcelona','马竞':'Ath Madrid','马德里竞技':'Ath Madrid',
    '塞维利亚':'Sevilla','贝蒂斯':'Betis','皇家社会':'Sociedad',
    '毕尔巴鄂':'Ath Bilbao','比利亚雷亚尔':'Villarreal','瓦伦西亚':'Valencia',
    '赫塔费':'Getafe','奥萨苏纳':'Osasuna','塞尔塔':'Celta',
    '拜仁':'Bayern Munich','拜仁慕尼黑':'Bayern Munich','多特蒙德':'Dortmund',
    '莱比锡':'RB Leipzig','莱红牛':'RB Leipzig','勒沃库森':'Leverkusen',
    '法兰克福':'Ein Frankfurt','斯图加特':'Stuttgart','门兴':"M'gladbach",
    '沃尔夫斯堡':'Wolfsburg','弗赖堡':'Freiburg','不来梅':'Werder Bremen',
    '国米':'Inter','国际米兰':'Inter','AC米兰':'Milan','米兰':'Milan',
    '尤文':'Juventus','尤文图斯':'Juventus','那不勒斯':'Napoli',
    '罗马':'Roma','拉齐奥':'Lazio','亚特兰大':'Atalanta',
    '佛罗伦萨':'Fiorentina','博洛尼亚':'Bologna','都灵':'Torino',
    '科莫':'Como','乌迪内斯':'Udinese','热那亚':'Genoa',
    '巴黎':'Paris SG','巴黎圣日耳曼':'Paris SG','马赛':'Marseille',
    '里昂':'Lyon','摩纳哥':'Monaco','里尔':'Lille','尼斯':'Nice',
    '朗斯':'Lens','雷恩':'Rennes','斯特拉斯堡':'Strasbourg',
}

def resolve_team(name, available):
    name = name.strip()
    target = TEAM_ALIASES.get(name, name)
    if target in available:
        return target
    matches = [t for t in available if t.casefold() == target.casefold()]
    return matches[0] if len(matches) == 1 else None

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

def extra_markets(lh, la):
    k = np.arange(15)
    pm = np.outer(poisson.pmf(k, lh), poisson.pmf(k, la))
    pm /= pm.sum()
    goals = np.add.outer(k, k)
    goal_probs = [float(pm[goals == i].sum()) for i in range(7)]
    goal_probs.append(float(pm[goals >= 7].sum()))
    # 假设两半场进球独立且期望值各占一半，非真实半场统计模型。
    half = np.outer(poisson.pmf(k, lh / 2), poisson.pmf(k, la / 2))
    half /= half.sum()
    names = ['胜', '平', '负']
    probs = {f'{a}{b}': 0.0 for a in names for b in names}
    for i in range(15):
        for j in range(15):
            first = 0 if i > j else (1 if i == j else 2)
            for x in range(15):
                for y in range(15):
                    full = 0 if i + x > j + y else (1 if i + x == j + y else 2)
                    probs[names[first] + names[full]] += float(half[i,j] * half[x,y])
    return goal_probs, probs

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

st.header("🎯 体彩每日分析 · 批量预测")
st.caption("支持中文球队简称和六个已接入联赛；自动按球队所属联赛匹配。跨联赛、日职、韩K联等暂不计算。")
st.write("每行一场，例如：001 阿森纳vs切尔西 -1（让球可省略，默认0）")
matches = st.text_area("比赛列表", placeholder="001 阿森纳vs切尔西 -1\n002 皇马vs巴萨 +1\n003 拜仁vs多特蒙德 -2", height=150)
if st.button("分析输入的比赛", type="primary"):
    import re
    rows = []
    datasets = {}
    for label, code in leagues.items():
        try:
            df = get_data(code, season)
            if len(df) >= 20:
                datasets[label] = (df, set(df.HomeTeam).union(df.AwayTeam))
        except Exception:
            pass
    for line in matches.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.fullmatch(r"(?:(\d{3})\s+)?(.+?)\s*(?:vs|VS|对阵|对)\s*(.+?)(?:\s+([+-]\d+))?\s*", line, flags=re.I)
        if not match:
            rows.append({'比赛':line, '状态':'格式错误：示例 001 阿森纳vs切尔西 -1'})
            continue
        number, home_name, away_name, hcap = match.groups()
        hcap = int(hcap or 0)
        choices = []
        for label, (df, available) in datasets.items():
            h = resolve_team(home_name, available)
            a = resolve_team(away_name, available)
            if h and a and h != a:
                choices.append((label, df, h, a))
        if not choices:
            rows.append({'比赛':line,'状态':'未匹配到同一已接入联赛的两支球队；请核对名称和赛季'})
            continue
        label, df, home, away = choices[0]
        rates = fit_rates(df, home, away)
        if rates is None:
            rows.append({'比赛':line,'联赛':label,'状态':'历史样本不足'})
            continue
        std, hcp, scores = probabilities(*rates, hcap)
        goal_probs, half_full = extra_markets(*rates)
        best_hf = sorted(half_full.items(), key=lambda x:-x[1])[:3]
        rows.append({'比赛':f'{number or "---"} {home_name.strip()} vs {away_name.strip()}',
                     '联赛':label,'让球':f'{hcap:+d}',
                     '胜':f'{std[0]:.1%}','平':f'{std[1]:.1%}','负':f'{std[2]:.1%}',
                     '让胜':f'{hcp[0]:.1%}','让平':f'{hcp[1]:.1%}','让负':f'{hcp[2]:.1%}',
                     '进球数概率': ' / '.join(f'{i if i < 7 else "7+"}球 {p:.0%}' for i,p in enumerate(goal_probs)),
                     '比分前三': '、'.join(f'{sc}({p:.1%})' for sc,p in scores[:3]),
                     '半全场前三': '、'.join(f'{sc}({p:.1%})' for sc,p in best_hf),
                     '状态':'历史基线估算'})
    if rows:
        output = pd.DataFrame(rows)
        st.dataframe(output, hide_index=True, use_container_width=True)
        st.download_button('下载分析结果 CSV',output.to_csv(index=False).encode('utf-8-sig'),
                           'daily_analysis.csv','text/csv')
    else:
        st.info('请先输入比赛。')
st.info('仅历史联赛数据；未接入官方体彩赔率、实时首发和伤停。半全场基于半场独立泊松近似，不是已验证的半场预测模型。')
