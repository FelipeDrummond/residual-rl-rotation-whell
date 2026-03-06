# Resumo da Pesquisa: RL Residual para Compensação de Stiction

## O Problema

Pêndulos invertidos com roda de reação sofrem com **atrito de Stribeck (stiction)** nos rolamentos. Em baixas velocidades angulares, o torque do motor comandado pelo LQR é insuficiente para mover a roda — ela trava, o pêndulo deriva, e o desempenho degrada. Mesmo com ganhos LQR ótimos (calculados via ARE contínua com back-EMF), o erro RMS sobe de **0.22° → 0.90°** — um problema não-linear que não se resolve com ajuste de ganhos.

## O Modelo de Atrito: Stribeck

O atrito de Stribeck modela três regimes:

```
|τ_atrito|
    Ts ─── ┐
            │╲
    Tc ─── │  ╲___________________  (nível Coulomb)
            │
            └──────────────────── |ω|
         0  vs
```

- **Stiction (|ω| ≈ 0):** τ = Ts = 0.15 Nm — a roda está **travada**. O torque motor precisa superar Ts para liberar.
- **Transição (|ω| ~ vs):** atrito cai exponencialmente de Ts para Tc.
- **Deslizamento (|ω| >> vs):** τ ≈ Tc = 0.09 Nm — atrito muito menor, roda gira livremente.

### Por que o stiction só importa no transiente

**Fase 1 (0–2s): Stiction é o gargalo.** A roda começa parada (α̇ = 0), bem na zona morta do stiction. O pêndulo está inclinado e o LQR comanda correções de tensão, mas em baixas velocidades angulares o torque motor é totalmente absorvido pelo atrito estático (Ts = 0.15 Nm). A roda permanece travada, o pêndulo deriva, e a recuperação é lenta. É aqui que o desempenho do LQR degrada de 0.22° para 0.90°.

**Fase 2 (2s+): Stiction é irrelevante.** Uma vez que a roda atinge |α̇| > ~1 rad/s, o atrito cai para o nível Coulomb (Tc = 0.09 Nm) e permanece baixo. O LQR tem autoridade total porque cada comando de tensão se traduz em torque real na roda.

### Stiction como freio natural

Quando o LQR leva θ → 0, seus comandos diminuem, e a roda eventualmente desacelera de volta para α̇ ≈ 0. Nesse ponto o stiction **trava** a roda novamente — e dessa vez isso é benéfico: o pêndulo fica perfeitamente no equilíbrio sustentado pelo atrito estático. O stiction age como um **freio gratuito** em regime permanente.

O papel do RL é ser um "kickstarter" — fornecer torque extra apenas na Fase 1 para vencer o stiction mais rápido, e depois desaparecer.

## A Solução

Arquitetura de controle híbrida com **gating angular**:

```
gate = min(1, |θ| / θ_threshold)
u_total = u_LQR + gate × α × π_θ(s)
```

- **LQR** estabiliza (dinâmica linearizada)
- **RL (PPO)** fornece torque suplementar para vencer o stiction durante transientes
- **Gate angular** (θ_threshold = 2.9°) faz a autoridade do RL diminuir suavemente conforme θ → 0, eliminando oscilações em regime permanente e permitindo que o stiction atue como freio natural

## Resultados (50 episódios, condições iniciais idênticas)

| Controlador | RMS θ | Transiente (0-2s) | Regime (8-10s) |
|---|---|---|---|
| LQR sem atrito | 0.22° | — | — |
| LQR com stiction | 0.90° | 1.21° | 0.000° |
| **Híbrido (LQR+RL)** | **0.48°** | **0.93°** | **0.003°** |

- **46.8% de melhoria** sobre o LQR com atrito
- **62% da lacuna fechada** entre desempenho com e sem atrito
- Chatter em regime permanente **completamente eliminado** (|u_RL| = 0.002V nos últimos 2s)

## Descoberta Chave: Back-EMF

O simulador inicialmente tinha Kv=0. Sem back-EMF, o RL aprendia um exploit de viés constante (mantinha a roda girando rápido para escapar do stiction). O amortecimento por back-EMF (Kt·Kv/Rm) é **10.6× mais forte** que o amortecimento linear — é o mecanismo dominante que limita a velocidade da roda no hardware real.

## Detalhes do Treinamento

- Algoritmo: PPO (Proximal Policy Optimization), Stable-Baselines3
- Observação: [θ, α, θ̇, α̇, u_RL_anterior] (5 dimensões)
- Ação: escalar ∈ [-1, 1], escalada para ±4V (autoridade limitada em ±12V)
- Rede: MLP compacta (para implantação no ESP32)
- Treinamento: 1M passos, 4 ambientes paralelos
- Gating angular: gate = min(1, |θ|/0.05) aplicado ao u_RL
- Recompensa: -(θ² + 0.1·θ̇² + 0.001·α̇² + 0.005·u_RL² + 0.005·Δu_RL²) + bônus(|θ|<0.1)

## Modelo de Simulação

Gêmeo digital com parâmetros da identificação de sistema no MATLAB:
- Pêndulo: Mh=0.149 kg, L=0.143 m, Jh=1.015e-3 kg·m²
- Roda: Mr=0.144 kg, Jr=1.317e-3 kg·m²
- Motor: 12V DC, Rm=6.67 Ω, Kt=0.174 Nm/A, Kv=0.285 V/(rad/s)
- Atrito de Stribeck: Ts=0.15 Nm, Tc=0.09 Nm, vs=0.02 rad/s
- Integração: RK4 com 10 sub-passos por período de controle de 20ms
- Ganhos LQR: K = [-45.0, 0.0, -5.2, -0.62] (via scipy ARE com back-EMF)

## Próximos Passos

1. Randomização de domínio para robustez sim-to-real
2. Exportação ONNX → C para inferência no ESP32 a 50Hz
3. Validação no pêndulo físico
