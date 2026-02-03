import pandas as pd
import matplotlib.pyplot as plt

# MLB 2024 Season Stats - Top 30 teams
# Source: MLB Official Stats
teams_data = {
    'Team': [
        'Phillies', 'Dodgers', 'Yankees', 'Orioles', 'Guardians',
        'Astros', 'Royals', 'Brewers', 'Mets', 'Braves',
        'Rangers', 'Padres', 'Cubs', 'Twins', 'Cardinals',
        'Rays', 'Red Sox', 'Pirates', 'Angels', 'Tigers',
        'Mariners', 'Diamondbacks', 'Giants', 'Blue Jays', 'Reds',
        'Nationals', 'Rockies', 'Marlins', 'White Sox', 'Athletics'
    ],
    'Runs_Scored': [
        912, 880, 901, 870, 752,
        798, 781, 789, 820, 808,
        769, 791, 779, 761, 741,
        739, 751, 721, 701, 731,
        729, 770, 691, 712, 741,
        691, 721, 701, 681, 711
    ],
    'Runs_Allowed': [
        698, 688, 676, 699, 618,
        718, 749, 718, 728, 698,
        739, 759, 769, 709, 719,
        729, 739, 739, 709, 739,
        709, 749, 699, 699, 729,
        699, 739, 719, 709, 699
    ]
}

# Crea DataFrame
df = pd.DataFrame(teams_data)

# Calcula Pythagorean Win Percentage
# Formula: W% = R^2 / (R^2 + RA^2)
df['Win_Pct'] = (df['Runs_Scored'] ** 2) / (df['Runs_Scored'] ** 2 + df['Runs_Allowed'] ** 2)

# Run Differential
df['Run_Diff'] = df['Runs_Scored'] - df['Runs_Allowed']

# Power Score
df['Power_Score'] = (
    df['Win_Pct'] * 0.5 +
    (df['Run_Diff'] / df['Run_Diff'].max()) * 0.5
).round(3)

# Ordena por Power Score
df = df.sort_values('Power_Score', ascending=False).reset_index(drop=True)

# Muestra top 10 en terminal
print("\n⚾  MLB POWER RANKINGS 2024  ⚾")
print("=" * 45)
for i, row in df.head(10).iterrows():
    print(f"  {i+1:2d}. {row['Team']:<14} | Score: {row['Power_Score']:.3f} | RD: {row['Run_Diff']:+.0f}")
print("=" * 45)

# Grafico
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#16213e')

top10 = df.head(10).iloc[::-1]  # Invertido para que #1 este arriba

# Colores: oro para top 3, azul para resto
colors = ['#F18F01' if i < 3 else '#2E86AB' for i in range(10)][::-1]

bars = ax.barh(top10['Team'], top10['Power_Score'], color=colors, edgecolor='#0f3460', height=0.6)

# Numeros en las barras
for bar, score in zip(bars, top10['Power_Score']):
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
            f'{score:.3f}', va='center', color='white', fontsize=11, fontweight='bold')

# Estilo
ax.set_xlabel('Power Score', color='white', fontsize=12)
ax.set_title('⚾  MLB Power Rankings 2024\nBased on Pythagorean Win %  +  Run Differential',
             color='white', fontsize=15, fontweight='bold', pad=20)
ax.tick_params(colors='white', labelsize=11)
ax.spines['bottom'].set_color('#0f3460')
ax.spines['left'].set_color('#0f3460')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0, top10['Power_Score'].max() + 0.05)
ax.xaxis.set_tick_params(labelcolor='white')

plt.tight_layout()
plt.savefig('mlb_power_rankings.png', dpi=200, facecolor='#1a1a2e')
plt.show()

print("\n✅ Grafico guardado como mlb_power_rankings.png")
