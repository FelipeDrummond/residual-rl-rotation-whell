# RL Residual para Compensação de Stiction em um Pêndulo Invertido com Roda de Reação

## Problema

Pêndulos invertidos com roda de reação sofrem com atrito de Stribeck (stiction) nos rolamentos da roda. Em baixas velocidades angulares, o atrito estático cria uma **zona morta** onde o torque motor comandado pelo controlador LQR é insuficiente para mover a roda. A roda permanece travada, o pêndulo deriva, e o desempenho do controle se degrada — mesmo com ganhos LQR otimamente ajustados.

No nosso sistema, o stiction (Ts=0.15 Nm) degrada o LQR ótimo de **0.22° RMS** (sem atrito) para **0.90° RMS** — um aumento de 4× no erro. Importante: isso não pode ser resolvido ajustando os ganhos LQR de forma mais agressiva — o problema é uma zona morta não-linear, não uma questão de ganho linear.

## Abordagem: Aprendizado por Reforço Residual

Utilizamos uma arquitetura de controle híbrida:

```
u_total(t) = u_LQR(t) + α · π_θ(s_t)
```

- **LQR** realiza a tarefa principal de estabilização (dinâmica linearizada)
- **RL (PPO)** aprende um torque suplementar pequeno (±4V de autoridade em ±12V) para superar o stiction

O agente RL é treinado em um gêmeo digital em Python (ambiente Gymnasium) utilizando Stable-Baselines3 PPO, com o objetivo de transferência sim-to-real para um microcontrolador ESP32.

## Modelo de Simulação

O gêmeo digital utiliza parâmetros físicos da identificação de sistema no MATLAB:
- Pêndulo: Mh=0.149 kg, L=0.143 m, Jh=1.015e-3 kg·m²
- Roda: Mr=0.144 kg, Jr=1.317e-3 kg·m²
- Motor: 12V DC, Rm=6.67 Ω, Kt=0.174 Nm/A, **Kv=0.285 V/(rad/s)**
- Atrito de Stribeck: Ts=0.15 Nm, Tc=0.09 Nm, vs=0.02 rad/s
- O atrito é propriamente acoplado às equações do pêndulo e da roda via a matriz de massa inversa (terceira lei de Newton)
- Integração: RK4 com 10 sub-passos por período de controle de 20ms

Ganhos LQR calculados via equação algébrica de Riccati contínua (scipy) para a planta linearizada **com força contra-eletromotriz (back-EMF)**: K = [-45.0, 0.0, -5.2, -0.62].

## Descoberta Chave: Back-EMF Era Crítico

As primeiras tentativas de treinamento falharam porque a simulação não incluía a **força contra-eletromotriz do motor** (Kv estava definido como 0). Sem back-EMF:
- A roda podia acelerar indefinidamente sob qualquer torque líquido
- O agente RL explorava isso aplicando um viés de tensão constante para manter a roda girando rápido, permanentemente fora da zona de stiction
- Nenhum ajuste de recompensa conseguia corrigir isso — a física do simulador estava errada

O amortecimento por back-EMF (Kt·Kv/Rm = 0.00744 Nm·s/rad) é **10.6× mais forte** que o amortecimento linear identificado (b2 = 0.000703 Nm·s/rad). É o mecanismo dominante que limita a velocidade da roda no hardware real. Uma vez adicionado, o exploit de viés constante desapareceu e o RL aprendeu compensação de stiction dependente do estado.

## Resultados

| Controlador | Erro RMS Angular | Observações |
|---|---|---|
| LQR ótimo (sem atrito) | 0.22° | Desempenho alvo |
| LQR ótimo (com stiction) | 0.90° | Degradado pela zona morta |
| **Híbrido (LQR + RL)** | **0.48°** | **Melhoria de 46.8%** |

### Análise Transiente vs Regime Permanente

| Métrica | LQR com Atrito | Híbrido (LQR+RL) |
|---|---|---|
| Primeiros 2s \|θ\| médio (transiente) | 1.21° | **0.93°** |
| Últimos 2s \|θ\| médio (regime permanente) | 0.000° | **0.003°** |
| Últimos 2s \|u_RL\| médio | — | 0.002V |

O agente RL treinado:
- Fecha **62% da lacuna** entre o desempenho degradado por atrito (0.90°) e o desempenho sem atrito (0.22°)
- **Elimina chatter em regime permanente** via gating angular (gate → 0 quando θ → 0)
- Atua somente durante transientes quando o pêndulo está longe do equilíbrio
- Mantém a velocidade da roda limitada (limite natural do back-EMF)

### Solução do Chatter: Gating Angular

Versões anteriores do controlador híbrido apresentavam oscilações de ±4V em regime permanente, pois o RL não tinha incentivo para parar de atuar quando θ ≈ 0. Resolvemos com um gate multiplicativo:

```
gate = min(1.0, |θ| / θ_threshold)    # θ_threshold = 0.05 rad (2.9°)
u_RL = gate × u_RL
```

Isso faz a autoridade do RL diminuir suavemente conforme o pêndulo se aproxima do equilíbrio, permitindo que o stiction atue como freio natural — idêntico ao comportamento do LQR puro em regime permanente.

## Detalhes do Treinamento

- Algoritmo: PPO (Proximal Policy Optimization)
- Observação: [θ, α, θ̇, α̇, u_RL_anterior] (5 dimensões)
- Ação: escalar ∈ [-1, 1], escalada para ±4V
- Rede: 32-32 política, 64-64 valor (compacta para implantação no ESP32)
- Treinamento: 1M passos, 4 ambientes paralelos, ent_coef=0.005, γ=0.995
- Recompensa: -(θ² + 0.1·θ̇² + 0.001·α̇² + 0.005·u_RL² + 0.005·Δu_RL²) + bônus(|θ|<0.1)
- Gating angular: gate = min(1, |θ|/0.05) aplicado ao u_RL para eliminar chatter em regime permanente

## Próximos Passos

1. **Randomização de domínio**: Reativar variação de ±10% em massas, atrito e inércias para robustez na transferência sim-to-real
2. **Exportação ONNX e implantação no ESP32**: Converter a MLP 32-32 para ponto fixo em C para inferência em tempo real a 50Hz
3. **Validação em hardware**: Testar no pêndulo físico para verificar a transferência sim-to-real
