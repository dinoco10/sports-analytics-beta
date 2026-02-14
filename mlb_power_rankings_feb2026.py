# ============================================================================
# MLB 2026 PRESEASON POWER RANKINGS - TU MODELO PROPIO
# ============================================================================
# Este script combina:
# 1. Proyecciones de múltiples fuentes (Steamer, ZiPS, ESPN)
# 2. Data de rosters actuales (MLB Stats API)
# 3. Underlying stats (WAR, rotación, bullpen, lineup)
# 4. TU INPUT PERSONAL (bias, eye test, conocimiento)
#
# El objetivo: crear un modelo único que refleje TU visión del baseball
# ============================================================================

"""
═══════════════════════════════════════════════════════════════════════════════
📖 CÓMO USAR ESTE ARCHIVO - GUÍA COMPLETA
═══════════════════════════════════════════════════════════════════════════════

PASO 1: PRIMER RUN (ver que funcione)
──────────────────────────────────────
Terminal/CMD:
    cd /ruta/donde/guardaste/este/archivo
    python mlb_power_rankings_FINAL.py

Esto va a:
✅ Validar que los weights suman 1.00
✅ Calcular rankings con valores default
✅ Generar gráfico: mlb_2026_power_rankings.png
✅ Generar CSV: mlb_2026_power_rankings.csv


PASO 2: AJUSTAR TUS WEIGHTS (tu filosofía de béisbol)
─────────────────────────────────────────────────────
Abrí este archivo en VSCode y andá a la función get_model_weights() (línea ~90)

EJEMPLO: "Creo que el bullpen es MÁS importante"

ANTES (líneas ~105-110):
    'bullpen_depth': 0.12,       # 12%
    'farm_system': 0.05,         # 5%

DESPUÉS:
    'bullpen_depth': 0.15,       # 15% ← Subiste +0.03
    'farm_system': 0.02,         # 2%  ← Bajaste -0.03
                                 # Total sigue siendo 1.00 ✅

TIPS:
- Si subís un weight, tenés que bajar otro(s) en la misma cantidad
- El código te va a AVISAR si no suman 1.00 (no se va a romper)
- Valores típicos: 0.05 (poco importante) a 0.20 (muy importante)


PASO 3: CALIFICAR EQUIPOS (componentes 1-10)
─────────────────────────────────────────────
Andá a la función get_team_components() (línea ~370)

ESCALA:
1-2   = Terrible, bottom 5
3-4   = Below average
5-6   = Average / median
7-8   = Above average / good
9-10  = Elite, top 5

EJEMPLO para Toronto Blue Jays:
    'rotation_strength': [
        10, 9, 8, ...  ← LAD, ATL, TOR (Toronto tiene rotación muy buena = 8)
    ],
    'bullpen_depth': [
        9, 8, 7, ...   ← LAD, ATL, TOR (bullpen sólido = 7)
    ],
    ...


PASO 4: TU BIAS PERSONAL (ajustes -10 a +10)
─────────────────────────────────────────────
Andá a la función get_personal_adjustments() (línea ~520)

ESCALA: -10 a +10 wins de ajuste

EJEMPLO: "Los Blue Jays van a ganar 3 juegos MÁS de lo que FanGraphs proyecta"
    'personal_adjustment': [
        0, 0, +3, ...  ← LAD, ATL, TOR (+3 para Toronto)
    ],
    'adjustment_reason': [
        '', '', 'Vladdy MVP year + pitching depth', ...
    ],


PASO 5: CORRER Y ANALIZAR
──────────────────────────
Terminal/CMD:
    python mlb_power_rankings_FINAL.py

Mirá los outputs:
📊 Terminal: Rankings y análisis
📈 PNG: Visualización de rankings
💾 CSV: Datos completos para Excel

Comparalo con FanGraphs:
- Si tu modelo da +5 wins vs FG → Sos más bullish en ese equipo
- Si tu modelo da -5 wins vs FG → Sos más bearish


PASO 6: ITERAR
───────────────
1. ¿El resultado te hace sentido?
   ✅ SI → Listo, tenés tu modelo
   ❌ NO → Ajustá weights / ratings / bias y corré de nuevo

2. Repetí hasta que el modelo refleje TU visión del baseball


═══════════════════════════════════════════════════════════════════════════════
🔧 OPCIONES AVANZADAS
═══════════════════════════════════════════════════════════════════════════════

ACTUALIZAR ROSTERS AUTOMÁTICAMENTE:
────────────────────────────────────
Los rosters cambian todos los días (trades, IL, call-ups)

Para actualizar antes de calcular rankings:
    Andá a la última línea del archivo (línea ~850)
    Cambiá:
        main(update_rosters=False)  ← ANTES
    Por:
        main(update_rosters=True)   ← DESPUÉS

Esto va a pull rosters actualizados de MLB Stats API.


USAR EN EXCEL:
──────────────
El CSV generado tiene TODAS las columnas:
- Tus ratings (1-10) para cada componente
- Tus ajustes personales
- FanGraphs proyecciones
- Rankings finales

Abrilo en Excel para hacer análisis adicional, filtros, gráficos, etc.


═══════════════════════════════════════════════════════════════════════════════
❓ PREGUNTAS FRECUENTES
═══════════════════════════════════════════════════════════════════════════════

P: ¿Qué pasa si los weights no suman 1.00?
R: El código te va a AVISAR con un error claro. No se va a romper,
   solo te dice qué ajustar. Seguí las instrucciones que aparecen.

P: ¿Puedo tener un weight de 0.00 (0%)?
R: Sí, significa que ese componente no te importa nada.

P: ¿Cuál es el weight máximo que puedo usar?
R: Técnicamente hasta 1.00 (100%), pero el código te va a advertir
   si algo es mayor a 0.30 (30%). Generalmente querés distribuir.

P: ¿Tengo que calificar TODOS los equipos?
R: Sí, pero podés usar 5 (average) para equipos que no conocés bien.

P: ¿Cada cuánto tengo que actualizar rosters?
R: Depende. Durante la temporada regular: diario. Pretemporada: semanal.

P: ¿Puedo usar este modelo durante la temporada?
R: Sí, solo tenés que actualizar:
   - Proyecciones de FanGraphs (cuando las publiquen)
   - Tus ratings de componentes (si cambiaron)
   - Tus ajustes personales (nuevas opiniones)

═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para solo guardar archivos
import matplotlib.pyplot as plt
import requests
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════
# ⚠️  SECCIÓN CRÍTICA: TUS WEIGHTS - MODIFICÁ SOLO ESTA PARTE ⚠️
# ═══════════════════════════════════════════════════════════════════════════
"""
🎯 CÓMO MODIFICAR LOS WEIGHTS SIN ROMPER NADA:

1. REGLA DE ORO: Los 10 componentes + personal_weight DEBEN sumar 1.00 (100%)

2. SI QUERÉS SUBIR UN WEIGHT:
   ✅ Subí el que te importa
   ✅ Bajá otro(s) en la misma cantidad
   ✅ Verificá que sigan sumando 1.00
   
   EJEMPLO:
   "Creo que el bullpen es MÁS importante que lo que tengo"
   
   ANTES:
   'bullpen_depth': 0.12,
   'farm_system': 0.05,
   
   DESPUÉS:
   'bullpen_depth': 0.15,  ← Subiste +0.03
   'farm_system': 0.02,    ← Bajaste -0.03
   Total sigue siendo 1.00 ✅

3. FILOSOFÍAS COMUNES:

   🎯 "Pitching wins championships"
   - Subir: rotation_strength (0.20), bullpen_depth (0.15)
   - Bajar: speed_baserunning (0.03), farm_system (0.03)
   
   🎯 "Offense is king"
   - Subir: lineup_power (0.20), lineup_contact (0.15)
   - Bajar: manager_coaching (0.05), momentum (0.05)
   
   🎯 "Trust my eye test"
   - Subir: personal_weight (0.20)
   - Bajar: otros proporcionalmente

4. EL CÓDIGO TE VA A AVISAR SI:
   ❌ Los weights no suman 1.00
   ❌ Algún weight es negativo
   ❌ Algún weight es mayor a 50%
   
   NO SE VA A ROMPER, solo te va a decir qué está mal.

5. TIPS:
   - Valores típicos: 0.05 (poco importante) a 0.20 (muy importante)
   - No todos pueden ser altos, tiene que haber balance
   - Empezá con los defaults y ajustá de a poco
"""

def get_model_weights():
    """
    ═══════════════════════════════════════════════════════════════════════
    🎯 TUS WEIGHTS - MODIFICÁ ESTOS NÚMEROS
    ═══════════════════════════════════════════════════════════════════════
    
    Cada número = % que pesa ese componente en tu modelo
    DEBEN SUMAR 1.00 (100%) en total
    """
    
    weights = {
        # ─────────────────────────────────────────────────────────────────
        # COMPONENTES DEL EQUIPO (calificás 1-10 cada uno)
        # ─────────────────────────────────────────────────────────────────
        # ⚾ AJUSTADO A TU FILOSOFÍA:
        # - Contact/OBP > Power
        # - Bullpen MUY importante
        # - Defense es plus
        # - Velocidad/Atletismo valorado
        # - Rotación: 3 buenos alcanza (no 5 aces)
        # - Manager: bajo (no sabés cómo gradear)
        
        'lineup_contact': 0.15,      # 15% - OBP/contacto > power (TU PRIORIDAD)
        'bullpen_depth': 0.15,       # 15% - MUY importante (ganar 2-1)
        'defense': 0.12,             # 12% - Plus importante
        'speed_baserunning': 0.10,   # 10% - Atletismo valorado
        'rotation_strength': 0.10,   # 10% - Medio (3 buenos alcanza)
        'lineup_power': 0.10,        # 10% - Medio (necesitás pero no énfasis)
        'depth': 0.08,               # 8%  - Sobrevivir 162 juegos
        'momentum': 0.07,            # 7%  - Vibes
        'farm_system': 0.05,         # 5%  - Call-ups
        'manager_coaching': 0.03,    # 3%  - Bajo (neutral, no sabés gradear)
        
        # ─────────────────────────────────────────────────────────────────
        # TU BIAS PERSONAL
        # ─────────────────────────────────────────────────────────────────
        
        'personal_weight': 0.05      # 5% - Empezás conservador, podés subir después
    }
    
    # ═════════════════════════════════════════════════════════════════════
    # VALIDACIÓN AUTOMÁTICA - NO TOCAR ESTA PARTE
    # ═════════════════════════════════════════════════════════════════════
    
    component_keys = [
        'rotation_strength', 'bullpen_depth', 'lineup_power', 'lineup_contact',
        'defense', 'speed_baserunning', 'depth', 'manager_coaching',
        'farm_system', 'momentum', 'personal_weight'
    ]
    
    total = sum(weights[k] for k in component_keys)
    
    print("\n" + "="*70)
    print("🔍 VALIDANDO TUS WEIGHTS")
    print("="*70)
    
    # Mostrar cada weight
    print("\n📊 TUS WEIGHTS:")
    for key in component_keys:
        print(f"   {key:20s}: {weights[key]:.2f} ({weights[key]*100:.0f}%)")
    
    print(f"\n{'─'*70}")
    print(f"   {'TOTAL':20s}: {total:.2f} ({total*100:.0f}%)")
    
    # Validar que suma 1.00
    if abs(total - 1.0) > 0.01:
        print("\n❌ ERROR: Los weights NO suman 1.00")
        print(f"   Suman: {total:.3f}")
        print(f"   Diferencia: {total - 1.0:+.3f}")
        print("\n💡 SOLUCIÓN:")
        print("   Ajustá los números arriba para que sumen exactamente 1.00")
        print("   Si subís uno, tenés que bajar otro en la misma cantidad")
        raise ValueError("❌ Weights no suman 1.00 - ajustá los valores")
    else:
        print("\n✅ VALIDACIÓN OK - Los weights suman 1.00")
    
    # Checkear negativos
    negative = {k: v for k, v in weights.items() if v < 0}
    if negative:
        print(f"\n❌ ERROR: Hay weights negativos: {negative}")
        raise ValueError("❌ Weights no pueden ser negativos")
    
    # Advertencia por weights muy altos
    high = {k: v for k, v in weights.items() if v > 0.30}
    if high:
        print(f"\n⚠️  ADVERTENCIA: Estos weights son muy altos (>30%): {high}")
        print("   Considerá distribuir mejor entre componentes")
    
    print("="*70 + "\n")
    
    return weights


# ═══════════════════════════════════════════════════════════════════════════
# CORRELACIONES ESTADÍSTICAS CON WINS
# ═══════════════════════════════════════════════════════════════════════════
"""
Estas correlaciones vienen de análisis empírico de MLB data.
Fuente: Correlación entre stats de equipo y wins totales.

Positivas: stat más alto = más wins
Negativas: stat más alto = menos wins (invertir cuando uses)
"""

STAT_CORRELATIONS = {
    # ─────────────────────────────────────────────────────────────────
    # PITCHING (defensa)
    # ─────────────────────────────────────────────────────────────────
    'ERA': -0.81,           # ERA bajo = bueno (por eso negativo)
    'FIP': -0.75,           # FIP bajo = bueno
    'WHIP': -0.80,          # WHIP bajo = bueno
    'IP': 0.80,             # Más innings = bueno (durabilidad)
    'K/BB': 0.74,           # Más strikeouts por walk = bueno
    'K/9': 0.59,            # Más K por 9 innings = bueno
    'BB/9': -0.69,          # Menos walks = bueno
    'HR/9': -0.52,          # Menos HRs = bueno
    
    # ─────────────────────────────────────────────────────────────────
    # BATTING (ofensa)
    # ─────────────────────────────────────────────────────────────────
    'wRC+': 0.78,           # wRC+ alto = bueno (100 = average)
    'OBP': 0.78,            # OBP alto = bueno
    'OPS': 0.73,            # OPS alto = bueno
    'OPS+': 0.78,           # OPS+ alto = bueno (100 = average)
    'SLG': 0.68,            # SLG alto = bueno
    'R': 0.78,              # Más runs anotados = bueno
    'RBI': 0.78,            # Más RBIs = bueno
    'HR': 0.57,             # Más HRs = bueno
    'BA': 0.59,             # Average alto = bueno
    'BB': 0.57,             # Más walks = bueno
}


# ═══════════════════════════════════════════════════════════════════════════
# TEAM STATS - DATOS DE EQUIPOS PARA CÁLCULO ESTADÍSTICO
# ═══════════════════════════════════════════════════════════════════════════
"""
FASE 1 (ACTUAL): Stats de equipo ingresadas manualmente
FASE 2 (FUTURO): Pull automático desde APIs (pybaseball, FanGraphs)

Para completar:
1. Buscá las stats proyectadas de cada equipo en FanGraphs Depth Charts
2. Completá los arrays con los valores
3. Durante la temporada, podés mezclar proyecciones con resultados actuales
"""

def get_team_stats_2026():
    """
    Stats proyectadas de equipos para 2026.
    
    IMPORTANTE: Estas son PROYECCIONES para pretemporada.
    Durante la temporada, mezclar con stats actuales usando ponderación temporal.
    
    Stats incluidas (con correlaciones):
    - wRC+ team: Ofensa normalizada (100 = average)
    - ERA: Earned run average (bajo = bueno)
    - FIP: Fielding Independent Pitching (bajo = bueno)
    - WHIP: Walks + Hits per IP (bajo = bueno)
    - OBP: On-base percentage (alto = bueno)
    
    FASE 2: Reemplazar esto con función que pull desde FanGraphs/pybaseball
    """
    
    stats = {
        'Team': [
            'Los Angeles Dodgers', 'Atlanta Braves', 'Toronto Blue Jays',
            'New York Yankees', 'Seattle Mariners', 'New York Mets',
            'Boston Red Sox', 'Philadelphia Phillies', 'Houston Astros',
            'Detroit Tigers', 'Baltimore Orioles', 'Chicago Cubs',
            'San Francisco Giants', 'Arizona Diamondbacks', 'Milwaukee Brewers',
            'Minnesota Twins', 'Cincinnati Reds', 'Kansas City Royals',
            'Tampa Bay Rays', 'Pittsburgh Pirates', 'San Diego Padres',
            'Texas Rangers', 'Cleveland Guardians', 'St. Louis Cardinals',
            'Los Angeles Angels', 'Washington Nationals', 'Oakland Athletics',
            'Chicago White Sox', 'Colorado Rockies', 'Miami Marlins'
        ],
        
        # ─────────────────────────────────────────────────────────────────
        # OFENSA (correlation: 0.78)
        # ─────────────────────────────────────────────────────────────────
        # wRC+ proyectado (100 = average, >100 = bueno)
        'wRC+': [
            115, 108, 110, 106, 105, 104,  # LAD, ATL, TOR, NYY, SEA, NYM
            109, 107, 102, 106, 103, 100,  # BOS, PHI, HOU, DET, BAL, CHC
            98, 99, 101, 97, 102, 99,      # SF, ARI, MIL, MIN, CIN, KC
            95, 96, 98, 97, 100, 96,       # TB, PIT, SD, TEX, CLE, STL
            94, 92, 88, 85, 87, 90         # LAA, WAS, OAK, CHW, COL, MIA
        ],
        
        # ─────────────────────────────────────────────────────────────────
        # PITCHING (correlation: -0.81, bajo = bueno)
        # ─────────────────────────────────────────────────────────────────
        # ERA proyectado
        'ERA': [
            3.45, 3.65, 3.70, 3.80, 3.60, 3.85,  # LAD, ATL, TOR, NYY, SEA, NYM
            3.75, 3.70, 3.90, 3.65, 3.85, 4.00,  # BOS, PHI, HOU, DET, BAL, CHC
            4.05, 4.10, 3.95, 4.15, 3.95, 4.05,  # SF, ARI, MIL, MIN, CIN, KC
            4.00, 4.20, 4.10, 4.25, 3.90, 4.15,  # TB, PIT, SD, TEX, CLE, STL
            4.40, 4.50, 4.65, 4.85, 4.70, 4.60   # LAA, WAS, OAK, CHW, COL, MIA
        ],
        
        # ─────────────────────────────────────────────────────────────────
        # FIP (correlation: -0.75, bajo = bueno)
        # ─────────────────────────────────────────────────────────────────
        # Fielding Independent Pitching proyectado
        'FIP': [
            3.50, 3.70, 3.75, 3.85, 3.65, 3.90,
            3.80, 3.75, 3.95, 3.70, 3.90, 4.05,
            4.10, 4.15, 4.00, 4.20, 4.00, 4.10,
            4.05, 4.25, 4.15, 4.30, 3.95, 4.20,
            4.45, 4.55, 4.70, 4.90, 4.75, 4.65
        ],
        
        # ─────────────────────────────────────────────────────────────────
        # WHIP (correlation: -0.80, bajo = bueno)
        # ─────────────────────────────────────────────────────────────────
        'WHIP': [
            1.15, 1.22, 1.23, 1.26, 1.20, 1.28,
            1.24, 1.23, 1.30, 1.22, 1.28, 1.33,
            1.35, 1.36, 1.31, 1.38, 1.31, 1.35,
            1.33, 1.40, 1.36, 1.42, 1.30, 1.38,
            1.46, 1.50, 1.55, 1.62, 1.57, 1.53
        ],
        
        # ─────────────────────────────────────────────────────────────────
        # OBP (correlation: 0.78, alto = bueno)
        # ─────────────────────────────────────────────────────────────────
        'OBP': [
            .340, .330, .335, .328, .327, .326,
            .332, .330, .322, .328, .325, .318,
            .315, .316, .320, .314, .320, .316,
            .312, .313, .315, .314, .318, .313,
            .310, .308, .302, .298, .300, .305
        ]
    }
    
    df = pd.DataFrame(stats)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# EXPECTED WINS - CÁLCULO USANDO CORRELACIONES
# ═══════════════════════════════════════════════════════════════════════════

def calculate_expected_wins_from_stats(team_stats_df):
    """
    Calcula wins esperados usando correlaciones estadísticas.
    
    Método: Regresión múltiple ponderada por correlaciones
    
    Fórmula:
    Expected Wins = 81 + Σ(stat_normalized × correlation × weight)
    
    Parámetros:
        team_stats_df: DataFrame con stats de equipos (wRC+, ERA, FIP, WHIP, OBP)
    
    Retorna:
        DataFrame con columna 'expected_wins' agregada
    """
    
    df = team_stats_df.copy()
    
    # Normalizar stats a escala -1 a +1 (para aplicar correlaciones)
    # Stats donde ALTO = BUENO (correlación positiva)
    df['wRC+_norm'] = (df['wRC+'] - 100) / 15  # Desviación de average (100)
    df['OBP_norm'] = (df['OBP'] - 0.320) / 0.020  # Desviación de .320
    
    # Stats donde BAJO = BUENO (correlación negativa) - invertir
    df['ERA_norm'] = -(df['ERA'] - 4.00) / 0.50  # Invertido
    df['FIP_norm'] = -(df['FIP'] - 4.00) / 0.50  # Invertido
    df['WHIP_norm'] = -(df['WHIP'] - 1.30) / 0.15  # Invertido
    
    # Aplicar correlaciones como pesos
    df['expected_wins'] = 81 + (
        df['wRC+_norm'] * STAT_CORRELATIONS['wRC+'] * 8 +
        df['OBP_norm'] * STAT_CORRELATIONS['OBP'] * 8 +
        df['ERA_norm'] * abs(STAT_CORRELATIONS['ERA']) * 8 +
        df['FIP_norm'] * abs(STAT_CORRELATIONS['FIP']) * 8 +
        df['WHIP_norm'] * abs(STAT_CORRELATIONS['WHIP']) * 8
    )
    
    # Clamp entre 50-110 wins (realista)
    df['expected_wins'] = df['expected_wins'].clip(50, 110)
    
    return df

# ============================================================================
# SECCIÓN 1: PROYECCIONES DE MÚLTIPLES FUENTES
# ============================================================================
# Compilamos las proyecciones de FanGraphs (Steamer/ZiPS), ESPN, etc.
# Esto nos da un "baseline" para comparar

def get_external_projections():
    """
    Proyecciones compiladas de múltiples fuentes para 2026.
    Fuentes: FanGraphs (Steamer/ZiPS), ESPN Power Rankings, MLB.com
    
    Columnas:
    - Team: nombre del equipo
    - FG_Wins: FanGraphs projected wins
    - FG_Playoff_Pct: FanGraphs playoff probability
    - FG_WS_Pct: FanGraphs World Series probability
    - ESPN_Rank: ESPN midwinter power ranking
    - Consensus_Tier: Tier basado en consenso (1=elite, 5=rebuild)
    """
    
    # Data compilada de las búsquedas web (febrero 2026)
    projections = {
        'Team': [
            'Los Angeles Dodgers', 'Atlanta Braves', 'Toronto Blue Jays', 
            'New York Yankees', 'Seattle Mariners', 'New York Mets',
            'Boston Red Sox', 'Philadelphia Phillies', 'Houston Astros',
            'Detroit Tigers', 'Baltimore Orioles', 'Chicago Cubs',
            'San Francisco Giants', 'Arizona Diamondbacks', 'Milwaukee Brewers',
            'Minnesota Twins', 'Cincinnati Reds', 'Kansas City Royals',
            'Tampa Bay Rays', 'Pittsburgh Pirates', 'San Diego Padres',
            'Texas Rangers', 'Cleveland Guardians', 'St. Louis Cardinals',
            'Los Angeles Angels', 'Washington Nationals', 'Oakland Athletics',
            'Chicago White Sox', 'Colorado Rockies', 'Miami Marlins'
        ],
        # FanGraphs Depth Charts projected wins (Feb 2026)
        'FG_Wins': [
            100, 92, 90, 88, 92, 89,
            90, 86, 85, 84, 84, 83,
            82, 81, 81, 80, 78, 79,
            78, 79, 79, 77, 75, 74,
            72, 70, 68, 65, 64, 62
        ],
        # FanGraphs playoff probability %
        'FG_Playoff_Pct': [
            99, 85, 80, 75, 82, 80,
            70, 70, 65, 61, 55, 50,
            45, 42, 48, 40, 35, 38,
            32, 35, 21, 25, 18, 15,
            10, 8, 5, 3, 2, 1
        ],
        # FanGraphs World Series probability %
        'FG_WS_Pct': [
            28.0, 10.2, 7.0, 5.5, 5.0, 6.1,
            4.0, 7.6, 6.8, 2.0, 2.0, 1.5,
            1.2, 1.0, 1.3, 0.8, 0.6, 0.7,
            0.5, 0.6, 0.7, 0.4, 0.3, 0.2,
            0.1, 0.1, 0.1, 0.0, 0.0, 0.0
        ],
        # ESPN Power Rankings position (1-30, lower = better)
        'ESPN_Rank': [
            1, 4, 2, 5, 3, 8,
            6, 9, 10, 7, 11, 13,
            14, 16, 12, 15, 19, 17,
            18, 20, 21, 22, 24, 23,
            25, 26, 27, 28, 29, 30
        ],
        # Consensus tier (1=contender, 2=playoff, 3=bubble, 4=rebuild, 5=tank)
        'Consensus_Tier': [
            1, 1, 1, 1, 1, 2,
            2, 2, 2, 2, 2, 3,
            3, 3, 3, 3, 3, 3,
            3, 3, 4, 4, 4, 4,
            4, 5, 5, 5, 5, 5
        ]
    }
    
    df = pd.DataFrame(projections)
    return df


# ============================================================================
# SECCIÓN 2: ROSTERS Y WAR PROYECTADO (MLB Stats API)
# ============================================================================

def get_team_rosters_2026():
    """
    Obtiene los rosters actuales de cada equipo vía MLB Stats API.
    Incluye 40-man roster con proyecciones de WAR donde estén disponibles.
    """
    
    # Primero obtenemos la lista de equipos
    teams_url = "https://statsapi.mlb.com/api/v1/teams?sportId=1&season=2026"
    
    try:
        response = requests.get(teams_url)
        data = response.json()
        
        teams_info = []
        for team in data['teams']:
            teams_info.append({
                'team_id': team['id'],
                'team_name': team['name'],
                'abbreviation': team.get('abbreviation', ''),
                'division': team['division']['name'],
                'league': team['league']['name']
            })
        
        return pd.DataFrame(teams_info)
    
    except Exception as e:
        print(f"Error fetching teams: {e}")
        return None


def get_team_roster_details(team_id):
    """
    Obtiene el roster detallado de un equipo específico.
    """
    roster_url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=40Man"
    
    try:
        response = requests.get(roster_url)
        data = response.json()
        
        players = []
        for player in data.get('roster', []):
            players.append({
                'player_id': player['person']['id'],
                'name': player['person']['fullName'],
                'position': player['position']['abbreviation'],
                'status': player.get('status', {}).get('description', 'Active')
            })
        
        return players
    
    except Exception as e:
        print(f"Error fetching roster: {e}")
        return []


def update_rosters_daily(save_to_file=True):
    """
    Actualiza todos los rosters desde MLB Stats API.
    Los rosters cambian todos los días (trades, IL moves, call-ups, etc.)
    
    Parámetros:
        save_to_file: Si True, guarda un CSV con timestamp
    
    Retorna:
        DataFrame con información de rosters actualizados
    """
    
    print("\n" + "="*70)
    print("🔄 ACTUALIZANDO ROSTERS DESDE MLB STATS API")
    print("="*70 + "\n")
    
    # Obtener todos los equipos
    teams_df = get_team_rosters_2026()
    
    if teams_df is None:
        print("❌ Error al obtener lista de equipos")
        return None
    
    rosters_summary = []
    
    for _, team in teams_df.iterrows():
        team_id = team['team_id']
        team_name = team['team_name']
        
        # Obtener roster del equipo
        roster = get_team_roster_details(team_id)
        
        if roster:
            # Contar por posición
            position_counts = {}
            for player in roster:
                pos = player['position']
                position_counts[pos] = position_counts.get(pos, 0) + 1
            
            # Contar injured
            injured_count = sum(1 for p in roster if p['status'] != 'Active')
            
            rosters_summary.append({
                'team': team_name,
                'roster_size': len(roster),
                'injured': injured_count,
                'pitchers': position_counts.get('P', 0),
                'position_players': len(roster) - position_counts.get('P', 0),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M')
            })
            
            print(f"✅ {team_name:25s} - {len(roster):2d} players ({injured_count} injured)")
        else:
            print(f"❌ {team_name:25s} - Error al obtener roster")
    
    # Crear DataFrame
    summary_df = pd.DataFrame(rosters_summary)
    
    if save_to_file and not summary_df.empty:
        filename = f"mlb_rosters_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        summary_df.to_csv(filename, index=False)
        print(f"\n💾 Rosters guardados en: {filename}")
    
    print("\n" + "="*70)
    print("✅ ROSTERS ACTUALIZADOS")
    print("="*70 + "\n")
    
    return summary_df


# ============================================================================
# SECCIÓN 3: UNDERLYING STATS - COMPONENTES DEL MODELO
# ============================================================================
# Estos son los factores que VOS vas a ponderar según tu criterio

def get_team_components():
    """
    Componentes subyacentes de cada equipo para 2026.
    Estos datos vienen de proyecciones + tu análisis.
    
    Escala 1-10 para cada componente:
    - rotation_strength: Calidad de la rotación titular
    - bullpen_depth: Profundidad y calidad del bullpen
    - lineup_power: Poder ofensivo del lineup
    - lineup_contact: Habilidad de contacto/OBP del lineup
    - defense: Calidad defensiva general
    - speed_baserunning: Velocidad y agresividad en bases
    - depth: Profundidad del roster (bench + minors)
    - manager_coaching: Calidad del manager y coaching staff
    - farm_system: Calidad del sistema de menores
    - momentum: Momentum/vibes entrando a 2026
    """
    
    # =========================================================================
    # IMPORTANTE: Estos valores son para que VOS los ajustes
    # Basate en tu conocimiento, en lo que leíste, en tu eye test
    # =========================================================================
    
    components = {
        'Team': [
            'Los Angeles Dodgers', 'Atlanta Braves', 'Toronto Blue Jays',
            'New York Yankees', 'Seattle Mariners', 'New York Mets',
            'Boston Red Sox', 'Philadelphia Phillies', 'Houston Astros',
            'Detroit Tigers', 'Baltimore Orioles', 'Chicago Cubs',
            'San Francisco Giants', 'Arizona Diamondbacks', 'Milwaukee Brewers',
            'Minnesota Twins', 'Cincinnati Reds', 'Kansas City Royals',
            'Tampa Bay Rays', 'Pittsburgh Pirates', 'San Diego Padres',
            'Texas Rangers', 'Cleveland Guardians', 'St. Louis Cardinals',
            'Los Angeles Angels', 'Washington Nationals', 'Oakland Athletics',
            'Chicago White Sox', 'Colorado Rockies', 'Miami Marlins'
        ],
        
        # ROTACIÓN TITULAR (1-10)
        # Considera: aces, depth, health, upside
        'rotation_strength': [
            10, 9, 8, 8, 9, 7,  # LAD, ATL, TOR, NYY, SEA, NYM
            8, 8, 7, 9, 7, 6,   # BOS, PHI, HOU, DET, BAL, CHC
            6, 5, 7, 6, 8, 6,   # SF, ARI, MIL, MIN, CIN, KC
            7, 6, 5, 5, 6, 5,   # TB, PIT, SD, TEX, CLE, STL
            4, 4, 3, 3, 3, 4    # LAA, WSH, OAK, CHW, COL, MIA
        ],
        
        # BULLPEN (1-10)
        # Considera: closer, setup men, depth, lefty options
        'bullpen_depth': [
            9, 8, 7, 7, 7, 7,
            7, 7, 7, 6, 6, 6,
            6, 5, 7, 6, 6, 7,
            8, 5, 6, 5, 7, 5,
            4, 4, 3, 3, 3, 4
        ],
        
        # PODER OFENSIVO (1-10)
        # Considera: HR, SLG, run production
        'lineup_power': [
            9, 9, 8, 9, 9, 8,
            8, 8, 8, 7, 7, 8,
            6, 7, 6, 6, 7, 6,
            6, 6, 7, 7, 6, 5,
            6, 4, 5, 4, 5, 4
        ],
        
        # CONTACTO/OBP (1-10)
        # Considera: AVG, OBP, K rate, plate discipline
        'lineup_contact': [
            8, 8, 8, 7, 7, 8,
            8, 7, 7, 6, 6, 7,
            7, 6, 7, 7, 6, 6,
            7, 6, 6, 6, 7, 5,
            5, 5, 4, 4, 4, 4
        ],
        
        # DEFENSA (1-10)
        # Considera: infield, outfield, catcher, overall
        'defense': [
            7, 8, 7, 7, 8, 6,
            8, 7, 7, 7, 8, 7,
            7, 7, 8, 6, 6, 7,
            8, 6, 6, 6, 8, 6,
            5, 5, 5, 4, 4, 5
        ],
        
        # VELOCIDAD/BASERUNNING (1-10)
        'speed_baserunning': [
            7, 8, 6, 6, 7, 6,
            7, 6, 6, 5, 7, 7,
            5, 7, 8, 6, 7, 7,
            8, 6, 6, 5, 6, 5,
            5, 5, 5, 4, 4, 5
        ],
        
        # PROFUNDIDAD DE ROSTER (1-10)
        'depth': [
            10, 8, 8, 8, 7, 7,
            7, 7, 7, 6, 7, 7,
            6, 6, 8, 6, 6, 6,
            8, 6, 5, 5, 6, 5,
            4, 5, 4, 3, 3, 4
        ],
        
        # MANAGER/COACHING (1-10)
        'manager_coaching': [
            10, 9, 8, 7, 8, 7,
            7, 8, 9, 7, 6, 7,
            7, 6, 9, 6, 6, 7,
            9, 6, 7, 7, 8, 6,
            5, 5, 5, 4, 4, 5
        ],
        
        # FARM SYSTEM (1-10)
        # Impacto futuro + call-ups potenciales en 2026
        'farm_system': [
            7, 7, 6, 6, 8, 6,
            8, 7, 5, 7, 9, 8,
            7, 7, 7, 7, 8, 7,
            9, 8, 6, 6, 7, 6,
            6, 8, 7, 7, 7, 7
        ],
        
        # MOMENTUM/VIBES (1-10)
        # ¿Cómo entra el equipo a 2026? ¿Confianza? ¿Expectativa?
        'momentum': [
            10, 7, 9, 7, 8, 6,
            7, 6, 6, 7, 5, 6,
            6, 5, 6, 5, 5, 6,
            5, 6, 4, 4, 4, 4,
            3, 4, 3, 2, 2, 3
        ]
    }
    
    df = pd.DataFrame(components)
    return df


# ============================================================================
# SECCIÓN 4: TU INPUT PERSONAL - EL DIFERENCIADOR
# ============================================================================

def get_personal_adjustments():
    """
    ACÁ ES DONDE VOS METÉS TU CONOCIMIENTO.
    
    Ajustes personales basados en:
    - Tu eye test
    - Cosas que sabés que los modelos no capturan
    - Lesiones que te preocupan
    - Jugadores que creés que van a explotar/declinar
    - Vibes que tenés sobre ciertos equipos
    
    Escala: -10 a +10 (ajuste a las wins proyectadas)
    
    Ejemplos de razones para ajustar:
    - +3: "Creo que Acuña va a tener un año MVP y los modelos no lo capturan"
    - -4: "El bullpen de este equipo me parece terrible, van a soplar leads"
    - +2: "Este manager es elite y siempre saca más del roster"
    - -3: "Este equipo tiene injury history y nadie lo menciona"
    """
    
    # =========================================================================
    # COMPLETÁ ESTO CON TUS AJUSTES PERSONALES
    # Poné 0 si no tenés opinión fuerte
    # =========================================================================
    
    adjustments = {
        'Team': [
            'Los Angeles Dodgers', 'Atlanta Braves', 'Toronto Blue Jays',
            'New York Yankees', 'Seattle Mariners', 'New York Mets',
            'Boston Red Sox', 'Philadelphia Phillies', 'Houston Astros',
            'Detroit Tigers', 'Baltimore Orioles', 'Chicago Cubs',
            'San Francisco Giants', 'Arizona Diamondbacks', 'Milwaukee Brewers',
            'Minnesota Twins', 'Cincinnati Reds', 'Kansas City Royals',
            'Tampa Bay Rays', 'Pittsburgh Pirates', 'San Diego Padres',
            'Texas Rangers', 'Cleveland Guardians', 'St. Louis Cardinals',
            'Los Angeles Angels', 'Washington Nationals', 'Oakland Athletics',
            'Chicago White Sox', 'Colorado Rockies', 'Miami Marlins'
        ],
        
        # TU AJUSTE EN WINS (-10 a +10)
        'personal_adjustment': [
            0, 0, 0, 0, 0, 0,   # Completá con tus ajustes
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0
        ],
        
        # RAZÓN DE TU AJUSTE (para que recuerdes por qué)
        'adjustment_reason': [
            '', '', '', '', '', '',
            '', '', '', '', '', '',
            '', '', '', '', '', '',
            '', '', '', '', '', '',
            '', '', '', '', '', ''
        ],
        
        # CONFIANZA EN TU AJUSTE (1-5, 5=muy seguro)
        'confidence': [
            3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3
        ]
    }
    
    df = pd.DataFrame(adjustments)
    return df


# ============================================================================
# SECCIÓN 5: EL MODELO - COMBINANDO TODO
# ============================================================================

def calculate_power_rankings(weights=None):
    """
    ═══════════════════════════════════════════════════════════════════════════
    MODELO - 3 FUENTES COMBINADAS
    ═══════════════════════════════════════════════════════════════════════════
    
    Combina 3 fuentes de información:
    
    1. FANGRAPHS (50%) - Baseline experto
       Proyecciones Steamer/ZiPS agregadas
    
    2. STATS-BASED (30%) - Correlaciones empíricas
       Expected wins calculados usando correlaciones estadísticas
       (wRC+, ERA, FIP, WHIP, OBP)
    
    3. TU CRITERIO (20%) - Tu conocimiento único
       - Componentes evaluados 1-10 (ponderados por tus weights)
       - Bias personal (-10 a +10)
    
    Parámetros:
        weights: diccionario con tus pesos de componentes
                 Si es None, usa get_model_weights() con validación
    
    Retorna:
        DataFrame con rankings finales y análisis completo
    """
    
    print("\n" + "="*80)
    print("🏗️  CONSTRUYENDO MODELO HÍBRIDO")
    print("="*80)
    
    # ═════════════════════════════════════════════════════════════════════════
    # PASO 1: Cargar weights y validar
    # ═════════════════════════════════════════════════════════════════════════
    if weights is None:
        weights = get_model_weights()
    
    # ═════════════════════════════════════════════════════════════════════════
    # PASO 2: Cargar todas las fuentes de datos
    # ═════════════════════════════════════════════════════════════════════════
    print("\n📊 Cargando fuentes de datos...")
    
    # Fuente 1: FanGraphs proyecciones
    projections = get_external_projections()
    print(f"   ✅ FanGraphs: {len(projections)} equipos")
    
    # Fuente 2: Stats de equipo (para correlaciones)
    team_stats = get_team_stats_2026()
    print(f"   ✅ Team Stats: {len(team_stats)} equipos")
    
    # Fuente 3a: Tus componentes evaluados
    components = get_team_components()
    print(f"   ✅ Tus componentes: {len(components)} equipos")
    
    # Fuente 3b: Tu bias personal
    personal = get_personal_adjustments()
    print(f"   ✅ Tus ajustes: {len(personal)} equipos")
    
    # ═════════════════════════════════════════════════════════════════════════
    # PASO 3: Calcular expected wins usando correlaciones
    # ═════════════════════════════════════════════════════════════════════════
    print("\n🔬 Calculando expected wins (correlaciones estadísticas)...")
    team_stats = calculate_expected_wins_from_stats(team_stats)
    print(f"   ✅ Expected wins calculados")
    
    # ═════════════════════════════════════════════════════════════════════════
    # PASO 4: Merge todo
    # ═════════════════════════════════════════════════════════════════════════
    df = projections.merge(team_stats[['Team', 'wRC+', 'ERA', 'FIP', 'WHIP', 'OBP', 'expected_wins']], 
                          on='Team', how='left')
    df = df.merge(components, on='Team', how='left')
    df = df.merge(personal, on='Team', how='left')
    
    # ═════════════════════════════════════════════════════════════════════════
    # PASO 5: Calcular component-based wins (tu evaluación)
    # ═════════════════════════════════════════════════════════════════════════
    print("\n🎯 Calculando wins basados en tus componentes...")
    
    # Normalizar componentes a escala 0-1
    component_cols = ['rotation_strength', 'bullpen_depth', 'lineup_power',
                      'lineup_contact', 'defense', 'speed_baserunning',
                      'depth', 'manager_coaching', 'farm_system', 'momentum']
    
    for col in component_cols:
        df[f'{col}_norm'] = df[col] / 10  # 1-10 → 0-1
    
    # Weighted average de componentes
    df['component_score'] = 0
    for col in component_cols:
        weight = weights.get(col, 0.1)
        df['component_score'] += df[f'{col}_norm'] * weight
    
    # Convertir score a wins (escala 60-100)
    min_wins = 60
    max_wins = 100
    df['component_wins'] = min_wins + (df['component_score'] * (max_wins - min_wins))
    
    # Aplicar tu bias personal
    personal_impact = df['personal_adjustment'] * 2  # Escalar -10/+10 a impacto en wins
    df['user_wins'] = df['component_wins'] + personal_impact
    
    print(f"   ✅ Wins basados en tu criterio calculados")
    
    # ═════════════════════════════════════════════════════════════════════════
    # PASO 6: MODELO FINAL - Combinar las 3 fuentes
    # ═════════════════════════════════════════════════════════════════════════
    print("\n🏆 Combinando las 3 fuentes en modelo final...")
    print("   50% FanGraphs (baseline experto)")
    print("   30% Stats-based (correlaciones empíricas)")
    print("   20% Tu criterio (componentes + bias)")
    
    # Pesos de las 3 fuentes
    FANGRAPHS_WEIGHT = 0.50
    STATS_WEIGHT = 0.30
    USER_WEIGHT = 0.20
    
    df['model_wins'] = (
        (df['FG_Wins'] * FANGRAPHS_WEIGHT) +
        (df['expected_wins'] * STATS_WEIGHT) +
        (df['user_wins'] * USER_WEIGHT)
    )
    
    # ═════════════════════════════════════════════════════════════════════════
    # PASO 7: Finalizar y calcular métricas
    # ═════════════════════════════════════════════════════════════════════════
    df['model_wins'] = df['model_wins'].round(1)
    df = df.sort_values('model_wins', ascending=False)
    df['rank'] = range(1, len(df) + 1)
    
    # Diferencias vs cada fuente
    df['vs_fangraphs'] = (df['model_wins'] - df['FG_Wins']).round(1)
    df['vs_stats'] = (df['model_wins'] - df['expected_wins']).round(1)
    df['vs_user'] = (df['model_wins'] - df['user_wins']).round(1)
    
    print(f"\n✅ MODELO COMPLETADO")
    print("="*80)
    
    return df


# ============================================================================
# SECCIÓN 6: VISUALIZACIÓN
# ============================================================================

def plot_power_rankings(df, show_comparison=True):
    """
    Visualiza los Power Rankings con comparación vs FanGraphs.
    Muestra TODOS los 30 equipos con diseño mejorado y 3 gráficos.
    """
    
    # Crear figura con 3 subplots
    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # ═══════════════════════════════════════════════════════════════════
    # GRÁFICO 1: TU MODELO VS FANGRAPHS (TODOS LOS 30 EQUIPOS)
    # ═══════════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0, :])  # Top, ocupa ambas columnas
    
    # Preparar datos - TODOS los equipos
    df_sorted = df.sort_values('model_wins', ascending=True).copy()
    teams_short = df_sorted['Team'].str.replace('Los Angeles ', 'LA ').str.replace('San Francisco', 'SF').str.replace('San Diego', 'SD')
    
    y_pos = np.arange(len(df_sorted))
    
    # Crear barras lado a lado
    bar_height = 0.35
    
    # FanGraphs (gris oscuro)
    bars1 = ax1.barh(y_pos - bar_height/2, df_sorted['FG_Wins'], 
                     bar_height, label='FanGraphs', 
                     color='#64748b', alpha=0.8, edgecolor='white', linewidth=0.5)
    
    # Tu Modelo (azul)
    bars2 = ax1.barh(y_pos + bar_height/2, df_sorted['model_wins'], 
                     bar_height, label='Tu Modelo',
                     color='#2563eb', alpha=0.95, edgecolor='white', linewidth=0.5)
    
    # Agregar valores en las barras
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        fg_val = df_sorted.iloc[i]['FG_Wins']
        model_val = df_sorted.iloc[i]['model_wins']
        
        # FanGraphs
        ax1.text(bar1.get_width() + 0.5, bar1.get_y() + bar1.get_height()/2,
                f'{fg_val:.0f}', va='center', fontsize=7, color='#475569', fontweight='bold')
        
        # Tu modelo
        ax1.text(bar2.get_width() + 0.5, bar2.get_y() + bar2.get_height()/2,
                f'{model_val:.1f}', va='center', fontsize=7, fontweight='bold', color='#1e40af')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(teams_short, fontsize=8)
    ax1.set_xlabel('Projected Wins', fontsize=13, fontweight='bold')
    ax1.set_title('MLB 2026 Power Rankings: Tu Modelo vs FanGraphs (30 Equipos)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlim(55, 105)
    ax1.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax1.grid(axis='x', alpha=0.25, linestyle='--', linewidth=0.5)
    ax1.axvline(x=81, color='#dc2626', linestyle='--', alpha=0.4, linewidth=1.5, label='.500 (81W)')
    
    # ═══════════════════════════════════════════════════════════════════
    # GRÁFICO 2: DIFERENCIAS (dónde diferís de FanGraphs)
    # ═══════════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[1, 0])
    
    df_diff = df.copy()
    df_diff['diff'] = df_diff['model_wins'] - df_diff['FG_Wins']
    df_diff = df_diff.sort_values('diff', ascending=True)
    
    # Colores: verde si sos más bullish, rojo si más bearish
    colors_diff = ['#dc2626' if x < -0.5 else '#22c55e' if x > 0.5 else '#94a3b8' for x in df_diff['diff']]
    
    teams_short_diff = df_diff['Team'].str.replace('Los Angeles ', 'LA ').str.replace('San Francisco', 'SF').str.replace('San Diego', 'SD')
    
    bars_diff = ax2.barh(range(len(df_diff)), df_diff['diff'], 
                        color=colors_diff, alpha=0.8, edgecolor='white', linewidth=0.5)
    
    # Agregar valores solo para diferencias significativas
    for i, (bar, val) in enumerate(zip(bars_diff, df_diff['diff'])):
        if abs(val) >= 1.0:  # Solo mostrar si diferencia >= 1 win
            label = f'{val:+.1f}'
            x_pos = val + (0.4 if val > 0 else -0.4)
            align = 'left' if val > 0 else 'right'
            ax2.text(x_pos, bar.get_y() + bar.get_height()/2,
                    label, va='center', ha=align, fontsize=6.5, fontweight='bold')
    
    ax2.set_yticks(range(len(df_diff)))
    ax2.set_yticklabels(teams_short_diff, fontsize=7)
    ax2.set_xlabel('Diferencia (Tu Modelo - FanGraphs)', fontsize=11, fontweight='bold')
    ax2.set_title('🔼 Más Bullish (verde) vs 🔽 Más Bearish (rojo)', 
                 fontsize=13, fontweight='bold', pad=15)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax2.grid(axis='x', alpha=0.25, linestyle='--', linewidth=0.5)
    ax2.set_xlim(-10, 10)
    
    # ═══════════════════════════════════════════════════════════════════
    # GRÁFICO 3: SCATTER PLOT (correlación)
    # ═══════════════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Color por tier
    tier_colors = {1: '#16a34a', 2: '#2563eb', 3: '#eab308', 4: '#f97316', 5: '#dc2626'}
    colors_scatter = [tier_colors.get(tier, '#94a3b8') for tier in df['Consensus_Tier']]
    
    scatter = ax3.scatter(df['FG_Wins'], df['model_wins'], 
                         c=colors_scatter, s=180, alpha=0.75, edgecolors='white', linewidth=2)
    
    # Línea de igualdad
    ax3.plot([55, 105], [55, 105], 'k--', alpha=0.3, linewidth=2.5, label='Igualdad')
    
    # Etiquetas para equipos con >3 wins de diferencia
    for _, row in df.iterrows():
        if abs(row['vs_fangraphs']) >= 3:
            team_label = row['Team'].split()[-1]  # Solo apellido
            ax3.annotate(team_label,
                       (row['FG_Wins'], row['model_wins']),
                       fontsize=8, alpha=0.95, fontweight='bold',
                       xytext=(6, 6), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
    
    ax3.set_xlabel('FanGraphs Projected Wins', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Tu Modelo Projected Wins', fontsize=12, fontweight='bold')
    ax3.set_title('Correlación: Tu Modelo vs FanGraphs', 
                 fontsize=13, fontweight='bold', pad=15)
    ax3.set_xlim(55, 105)
    ax3.set_ylim(55, 105)
    ax3.grid(alpha=0.25, linestyle='--', linewidth=0.5)
    
    # Legend para tiers
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#16a34a', label='Tier 1: Contenders', edgecolor='white', linewidth=1),
        Patch(facecolor='#2563eb', label='Tier 2: Playoff', edgecolor='white', linewidth=1),
        Patch(facecolor='#eab308', label='Tier 3: Bubble', edgecolor='white', linewidth=1),
        Patch(facecolor='#f97316', label='Tier 4: Rebuild', edgecolor='white', linewidth=1),
        Patch(facecolor='#dc2626', label='Tier 5: Tank', edgecolor='white', linewidth=1)
    ]
    ax3.legend(handles=legend_elements, loc='lower right', fontsize=8.5, framealpha=0.95)
    
    # Ajustar layout y guardar
    plt.tight_layout()
    plt.savefig('mlb_2026_power_rankings.png', dpi=220, bbox_inches='tight', facecolor='white')
    print("\n✅ Gráfico guardado: mlb_2026_power_rankings.png")
    print("   📊 Gráfico 1: Tu Modelo vs FanGraphs (30 equipos comparados)")
    print("   📊 Gráfico 2: Diferencias (bullish/bearish)")
    print("   📊 Gráfico 3: Scatter plot con tiers")
    plt.close('all')  # Liberar memoria


def print_detailed_rankings(df):
    """
    Imprime los rankings con análisis detallado del MODELO HÍBRIDO.
    Muestra breakdown de las 3 fuentes.
    """
    
    print("\n" + "="*80)
    print("🏆 MLB 2026 POWER RANKINGS - MODELO HÍBRIDO")
    print("="*80)
    
    print("\n📊 TOP 10 EQUIPOS:\n")
    
    for _, row in df.head(10).iterrows():
        diff = row['vs_fangraphs']
        diff_str = f"+{diff:.1f}" if diff > 0 else f"{diff:.1f}"
        arrow = "🔼" if diff > 0 else "🔽" if diff < 0 else "➡️"
        
        print(f"#{int(row['rank']):2d} {row['Team']}")
        print(f"    🎯 MODELO FINAL: {row['model_wins']:.1f} wins")
        print(f"       └─ FanGraphs: {row['FG_Wins']:.0f}W | Stats: {row['expected_wins']:.1f}W | Tu Criterio: {row['user_wins']:.1f}W")
        print(f"    📊 Team Stats: wRC+ {row['wRC+']:.0f} | ERA {row['ERA']:.2f} | FIP {row['FIP']:.2f}")
        print(f"    vs FanGraphs: {diff_str} {arrow}")
        if row.get('adjustment_reason') and row['adjustment_reason']:
            print(f"    💭 Tu nota: {row['adjustment_reason']}")
        print()
    
    print("\n" + "="*80)
    print("📈 ANÁLISIS DE DIFERENCIAS:")
    print("="*80)
    
    # Equipos donde sos más bullish
    bullish = df.nlargest(5, 'vs_fangraphs')
    print("\n🔼 Más BULLISH que FanGraphs:")
    for _, row in bullish.iterrows():
        if row['vs_fangraphs'] > 0:
            print(f"   {row['Team']:30s}: +{row['vs_fangraphs']:4.1f} wins")
            print(f"      Stats dice: {row['expected_wins']:.1f}W | Tu criterio: {row['user_wins']:.1f}W")
    
    # Equipos donde sos más bearish
    bearish = df.nsmallest(5, 'vs_fangraphs')
    print("\n🔽 Más BEARISH que FanGraphs:")
    for _, row in bearish.iterrows():
        if row['vs_fangraphs'] < 0:
            print(f"   {row['Team']:30s}: {row['vs_fangraphs']:4.1f} wins")
            print(f"      Stats dice: {row['expected_wins']:.1f}W | Tu criterio: {row['user_wins']:.1f}W")
    
    print("\n" + "="*80)
    print("🔬 BREAKDOWN POR FUENTE:")
    print("="*80)
    
    # Correlación entre fuentes
    fg_stats_corr = df['FG_Wins'].corr(df['expected_wins'])
    fg_user_corr = df['FG_Wins'].corr(df['user_wins'])
    stats_user_corr = df['expected_wins'].corr(df['user_wins'])
    
    print(f"\n📊 Correlaciones entre fuentes:")
    print(f"   FanGraphs ↔ Stats:      {fg_stats_corr:.3f}")
    print(f"   FanGraphs ↔ Tu Criterio: {fg_user_corr:.3f}")
    print(f"   Stats ↔ Tu Criterio:     {stats_user_corr:.3f}")
    
    # Stats summary
    print(f"\n📈 Rango de proyecciones:")
    print(f"   FanGraphs:   {df['FG_Wins'].min():.0f}W - {df['FG_Wins'].max():.0f}W")
    print(f"   Stats-based: {df['expected_wins'].min():.1f}W - {df['expected_wins'].max():.1f}W")
    print(f"   Tu Criterio: {df['user_wins'].min():.1f}W - {df['user_wins'].max():.1f}W")
    print(f"   MODELO FINAL: {df['model_wins'].min():.1f}W - {df['model_wins'].max():.1f}W")
    
    # Equipos donde sos más bearish
    bearish = df.nsmallest(5, 'vs_fangraphs')
    print("\n🔽 Más BEARISH que FanGraphs:")
    for _, row in bearish.iterrows():
        if row['vs_fangraphs'] < 0:
            print(f"   {row['Team']}: {row['vs_fangraphs']:.1f} wins vs FG")


# ============================================================================
# SECCIÓN 7: EXPORTAR DATOS
# ============================================================================

def export_to_csv(df, filename='mlb_2026_power_rankings.csv'):
    """
    Exporta los rankings a CSV para análisis adicional.
    """
    
    export_cols = [
        'rank', 'Team', 'model_wins', 'FG_Wins', 'vs_fangraphs',
        'FG_Playoff_Pct', 'FG_WS_Pct', 'Consensus_Tier',
        'rotation_strength', 'bullpen_depth', 'lineup_power',
        'lineup_contact', 'defense', 'speed_baserunning',
        'depth', 'manager_coaching', 'farm_system', 'momentum',
        'personal_adjustment', 'adjustment_reason', 'confidence'
    ]
    
    df[export_cols].to_csv(filename, index=False)
    print(f"\n💾 Datos exportados a: {filename}")



# ═══════════════════════════════════════════════════════════════════════════
# 🚀 FASE 2: ROADMAP PARA MODELO BOTTOM-UP COMPLETO
# ═══════════════════════════════════════════════════════════════════════════
"""
Este es el modelo FASE 1 (híbrido).
Cuando estés listo para FASE 2, seguí estos pasos:

═══════════════════════════════════════════════════════════════════════════
PASO 1: Datos de jugadores individuales
═══════════════════════════════════════════════════════════════════════════

Instalar pybaseball:
    pip install pybaseball

Función para pull de stats proyectadas:

    from pybaseball import steamer_pitchers, steamer_hitters
    
    def get_player_projections_2026():
        # Pull Steamer projections
        pitchers = steamer_pitchers(season=2026)
        hitters = steamer_hitters(season=2026)
        
        # Agregar team affiliation
        # Calcular stats por equipo
        return team_aggregated_stats

═══════════════════════════════════════════════════════════════════════════
PASO 2: Agregación jugador → equipo
═══════════════════════════════════════════════════════════════════════════

Función para agregar:

    def aggregate_team_stats(player_projections):
        team_stats = {}
        
        for team in teams:
            # HITTING
            team_hitters = player_projections[hitters for team]
            team_wRC+ = weighted_average(
                values=team_hitters['wRC+'],
                weights=team_hitters['PA']  # Weighted by PA
            )
            
            # PITCHING
            team_pitchers = player_projections[pitchers for team]
            team_ERA = weighted_average(
                values=team_pitchers['ERA'],
                weights=team_pitchers['IP']  # Weighted by IP
            )
            
            team_stats[team] = {
                'wRC+': team_wRC+,
                'ERA': team_ERA,
                ...
            }
        
        return team_stats

═══════════════════════════════════════════════════════════════════════════
PASO 3: Ponderación temporal (proyecciones → actual)
═══════════════════════════════════════════════════════════════════════════

Función para mezclar proyecciones con resultados actuales:

    def blend_projections_with_actual(games_played):
        # Ponderación según games played
        if games_played < 10:
            weight_projections = 0.90
        elif games_played < 30:
            weight_projections = 0.70
        elif games_played < 60:
            weight_projections = 0.50
        elif games_played < 100:
            weight_projections = 0.30
        else:
            weight_projections = 0.10
        
        weight_actual = 1 - weight_projections
        
        # Mezclar
        blended_stat = (
            (actual_stat * weight_actual) +
            (projected_stat * weight_projections)
        )
        
        return blended_stat

═══════════════════════════════════════════════════════════════════════════
PASO 4: Actualización automática durante temporada
═══════════════════════════════════════════════════════════════════════════

Setup de cron job para actualizar diariamente:

    # Agregar a crontab:
    0 6 * * * cd /path/to/project && python update_rankings.py

Script update_rankings.py:

    def update_rankings():
        # 1. Pull actual stats (MLB Stats API)
        # 2. Blend con proyecciones
        # 3. Recalcular rankings
        # 4. Update gráficos
        # 5. Tweet/post automático

═══════════════════════════════════════════════════════════════════════════
PASO 5: Dashboard interactivo (opcional)
═══════════════════════════════════════════════════════════════════════════

Usar Streamlit para dashboard live:

    pip install streamlit
    
    import streamlit as st
    
    st.title("MLB Power Rankings - Live")
    
    # Selector de fecha
    date = st.date_input("Fecha")
    
    # Update on-demand
    if st.button("Actualizar"):
        rankings = calculate_power_rankings()
        st.plotly_chart(rankings_chart)

═══════════════════════════════════════════════════════════════════════════
"""


# ═══════════════════════════════════════════════════════════════════════════
# MAIN - EJECUTAR TODO
# ═══════════════════════════════════════════════════════════════════════════

def main(update_rosters=False):
    """
    Ejecuta el análisis completo.
    
    Parámetros:
        update_rosters: Si True, actualiza rosters desde MLB API antes de calcular
    """
    
    print("="*80)
    print("⚾ MLB 2026 POWER RANKINGS - MODELO HÍBRIDO")
    print("   Fase 1: Híbrido (FanGraphs + Stats + Tu Criterio)")
    print("="*80)
    
    # Opcionalmente actualizar rosters
    if update_rosters:
        update_rosters_daily()
    
    # Calcular rankings (esto va a validar tus weights automáticamente)
    df = calculate_power_rankings()
    
    # Mostrar resultados
    print_detailed_rankings(df)
    
    # Visualizar
    print("\n📈 Generando visualizaciones...")
    plot_power_rankings(df)
    
    # Exportar
    export_to_csv(df)
    
    print("\n" + "="*80)
    print("✅ ANÁLISIS COMPLETADO")
    print("="*80)
    print("\n💡 PRÓXIMOS PASOS (FASE 1):")
    print("   1. Completá team stats en get_team_stats_2026()")
    print("   2. Ajustá weights en get_model_weights() según tu filosofía")
    print("   3. Calificá equipos en get_team_components() (1-10)")
    print("   4. Agregá tus ajustes en get_personal_adjustments() (-10 a +10)")
    print("   5. Corré de nuevo y analiza resultados")
    print("\n🚀 FUTURO (FASE 2):")
    print("   Ver comentarios al final del código para roadmap completo")
    print("   Incluye: pull automático de jugadores, ponderación temporal, etc.")
    print("\n💾 OUTPUTS GENERADOS:")
    print("   - mlb_2026_power_rankings.png (gráficos mejorados)")
    print("   - mlb_2026_power_rankings.csv (datos completos)")
    if update_rosters:
        print("   - mlb_rosters_YYYYMMDD_HHMM.csv (rosters actualizados)")


if __name__ == "__main__":
    # Para actualizar rosters antes de calcular, cambiá a True
    # main(update_rosters=True)
    main(update_rosters=False)