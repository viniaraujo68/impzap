# Truco Paulista RL Environment

Este repositório contém um ambiente de Aprendizado por Reforço (Reinforcement Learning) focado no jogo de cartas de informação imperfeita Truco Paulista. 

O projeto adota uma arquitetura híbrida de alto desempenho: o motor de regras do jogo e a Máquina de Estados Finita (FSM) são construídos em **Go**, enquanto o ambiente de simulação e os agentes de IA são desenvolvidos em **Python** utilizando a interface padrão do OpenAI Gymnasium.

## Arquitetura do Monorepo

* `engine/`: Contém o código-fonte em Go (`truco.go`). É compilado via CGO para uma biblioteca compartilhada C (`.so`). Responsável por garantir 100% de segurança de memória e velocidade extrema no self-play.
* `env/`: O wrapper em Python (`truco_env.py`) que encapsula o binário C em uma classe `gym.Env` limpa, expondo métodos como `reset()` e `step()`.
* `agents/`: Os agentes que interagem com o ambiente. Inclui baselines (Random, Heuristic) e os modelos de aprendizado (REINFORCE, CFR).
* `notebooks/`: Análises experimentais e visualizações de treinamento.

## Requisitos

* Go 1.20+
* Python 3.10+
* GCC (para a compilação CGO)

## Instalação e Configuração

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/viniaraujo68/impzap.git](https://github.com/viniaraujo68/impzap.git)
   cd impzap