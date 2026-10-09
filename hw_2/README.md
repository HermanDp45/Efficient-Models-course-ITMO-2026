# HW2: Cut Cross-Entropy

[Cut Your Losses in Large-Vocabulary Language Models](https://arxiv.org/abs/2411.09009) (Wijmans et al., Apple, ICLR 2025)

CCE считает cross-entropy и её градиент, не записывая матрицу логитов `N × |V|` в глобальную память GPU: логиты считаются блоками в SRAM, наружу уходят только векторы длины `N`. За счёт этого память на лосс падает с `O(N·|V|)` до `O(N + |V|)`. Для замеров взят код авторов: [apple/ml-cross-entropy](https://github.com/apple/ml-cross-entropy).

## Структура

```text
hw_2/
├── reproduce.ipynb          # запуск бенчмарка + своя LSTM-модель с CCE3
├── setup.md                 # мои команды установки окружения
├── out/
│   ├── gemma2.csv           # сырые результаты бенчмарка, Gemma 2 (2B)
│   └── llama3_2_3.csv       # то же для Llama 3.2 (3B)
├── ml-cross-entropy/        # код авторов (коммит 3de376c), без изменений
│   ├── cut_cross_entropy/   #   Triton-ядра CCE и обёртка linear_cross_entropy
│   └── benchmark/           #   бенчмарк из Table 1 статьи
├── paper/
│   └── 2411.09009v2.pdf
└── slides/
    ├── Eff_nn_hw_2.pptx     # доклад, первая версия
    └── Eff_nn_hw_2_v2.pptx  # доклад, расширенная версия
```

## Окружение

- 2 × **NVIDIA A100 80GB PCIe**: на одной бенчмарк прогоняет модель, чтобы получить реальные эмбеддинги `E`, на другой идут замеры. Есть некоторый байес, посколько гпу были заняты не только мной, поэтому значения не идеально чистые, однако в сулчае их бенчмарка используется несколько сидов, а в случае моих отдельных запусков еще и среднее по нескольким прогонам для объективности.
- PyTorch **2.6.0+cu124**, Triton **3.2**, Python 3.10. Вот Тритон надо новее, однако не мог из-за драйверах на видюхах, но результат хороший, потому принимаю в допущения, что ничего не сломалось.
- bf16, батч **8 192** токена из Alpaca, `n_iteration=50`, `n_rep=1`

В статье Table 1 мерили на A100-SXM4 80GB, PyTorch 2.4.1, CUDA 12.4, с усреднением по 5 сидам.

## Результаты

Методы: `baseline` (обычный PyTorch), `torch_compile`, `cce`, а также точные варианты: `cce_kahan_full_c` (без фильтрации градиента по `C`, fp32-аккумуляция) и `cce_exact` (без фильтрации вообще, fp32-аккумуляция).

### Gemma 2 (2B): сравнение с Table 1

`|V| = 256 000`, `D = 2 304`. В каждой ячейке: **мой замер** / статья.

| Метод | Loss, МБ | Loss, мс | Gradient, МБ | Gradient, мс | Loss+Grad, МБ | Loss+Grad, мс |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | **24 000** / 24 000 | **86** / 82 | **16 000** / 16 000 | **129** / 122 | **28 000** / 28 000 | **225** / 208 |
| torch.compile | **4 000** / 4 000 | **51** / 49 | **1 161** / 12 000 | **89** / 92 | **5 161** / 16 000 | **145** / 143 |
| CCE | **1** / 1 | **80** / 46 | **1 162** / 1 163 | **101** / 100 | **1 163** / 1 164 | **185** / 145 |
| CCE-Kahan-FullC | **1** / 1 | **80** / 47 | **3 448** / 2 326 | **308** / 268 | **3 449** / 2 326 | **392** / 313 |
| CCE exact | **0.09** / — | **75** / — | **3 447** / — | **419** / — | **3 447** / — | **498** / — |

Нижняя граница памяти на градиент равна размеру самих `∇E` и `∇C` в bf16, это 1 161 МБ.

- **Память.** По памяти все отлично и полностью подтвердилось, CCE держится минимальным и лучше всех – особенно это как раз-таки заментно в сравнении с бейзлайном, 24 гига и 1 мегабайт - просто поразительно. Однако удивительно, что `torch.compile` получился настолько лучше, чем предполагалось.
  
- **Время.** тут значения достаточно похожы, но тут, во-первых, стохастика есть, а, во-вторых, еще и то, что карты были загружены не только мной.
  


### Llama 3.2 (3B): модель, которую они в статье не приводили

`|V| = 128 256`, `D = 3 072`. Нижняя граница памяти 799.5 МБ.

| Метод | Loss, МБ | Loss, мс | Gradient, МБ | Gradient, мс | Loss+Grad, МБ | Loss+Grad, мс |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 10 020 | 83 | 8 016 | 131 | 12 024 | 218 |
| torch.compile | 2 004 | 57 | 799 | 101 | 2 804 | 165 |
| **CCE** | **0.6** | **61** | **800** | **149** | **801** | **204** |
| CCE-Kahan-FullC | 0.6 | 65 | 2 352 | 448 | 2 353 | 501 |
| CCE exact | 0.09 | 63 | 2 352 | 535 | 2 352 | 593 |

По памяти получается также хорошо. CCE в ~15 раз меньше baseline. По времени CCE на 24% медленнее torch.compile. Это согласуется со статьёй: чем меньше отношение `|V| / D` (здесь = ~42, у Gemma 2 2B = ~111), тем меньше выигрыш CCE во времени. Для Phi 3.5 при `|V| / D = 10` авторы получили +50%.

### Своя модель: LSTM-LM с CCE

Тут я уже встроил прям отдельно в врзитектуру, написанную на торче. Единственное, реализация немного игрушечная - forward проход сразу включает в себя подсчет лосса, чтобы веса были одинаковые и результаты более объективные:
```python
class RNN(nn.Module):
    def __init__(self, vocab, embedding=256, hidden=512, layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab, embedding)
        self.rnn = nn.LSTM(embedding, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, vocab, bias=False)
    
    def encode(self, x):
        h = self.embedding(x)
        h, _ = self.rnn(h)
        return h

    def forward_baseline(self, x, targets):
        h = self.encode(x)
        logits = self.head(h)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )

    def forward_cce(self, x, targets):
        h = self.encode(x)
        return linear_cross_entropy(
            h.reshape(-1, h.size(-1)),
            self.head.weight,
            targets.reshape(-1)
        )
```
Стоило выбрать достаточно жирную конфигурацию + bf16, чтобы было объективно использовать.
Стоит отметить, что здесь память целиком на модель, а не только на лосс, но так даже лучше видно общий вклад (но сама модель не содержит всяких оптимизации, только вот compile)

Параметры:
Embedding 256 → LSTM 2 × 512 → Linear на `|V| = 128 000`, `B = 8`, `T = 512`, bf16. Голова `lm_head + F.cross_entropy` заменена одним вызовом `linear_cross_entropy(h, head.weight, targets)`.

| Вариант | fwd, мс | bwd, мс | fwd+bwd, мс | fwd, МиБ | bwd, МиБ | fwd+bwd, МиБ |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 65.1 | 181.3 | 254.8 | 2 338 | 3 528 | 3 334 |
| baseline + compile | 65.9 | 182.4 | 253.0 | 1 338 | 2 658 | 2 464 |
| **CCE** | **66.3** | **183.6** | **252.6** | **338** | **660** | **539** |

Пиковая память на шаг меньше в 6.2 раза, что в очередной раз подтвердило полезность данной реализации.

### Выводы

Статья меня крайне шокировала, а итоговый результат окончательно добил, а не какие-то копейки, что порадовало вдвойне. Простота интеграции в решение + то, что можно реализовать самостоятельно на тритоне - второй нереальный плюс. 

Сами результаты, соответственно, удалось воспроизвести, а также применить отдельно: память на лосс падает почти до нуля, по времени CCE на уровне baseline и немного уступает torch.compile, и отставание растёт при уменьшении `|V| / D`. Точные варианты (`kahan_full_c`, `exact`) обходятся в 3–4 раза более долгим backward и втрое большей памятью, но авторы рекомендуют их для pretraining.

## Как запустить

Нужна NVIDIA GPU с поддержкой Triton. Бенчмарк загружает модель на соседнюю GPU (`(cur + 1) % device_count`), на одной GPU он тоже запустится.

**1. Окружение.** Полный список моих команд лежит в [`setup.md`](setup.md), основное:

```bash
conda create -n cce_bench python=3.10 -y && conda activate cce_bench
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

cd hw_2/ml-cross-entropy
pip install -e ".[all]"
# flash-attn нужен для Llama (у Gemma 2 eager attention); колесо под torch 2.6 / cu12 / py3.10
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

Чтобы не ловить предупреждение про Triton 3.2, лучше поставить PyTorch ≥ 2.7, но мои замеры сделаны на 2.6.

**2. Доступ к моделям.** `google/gemma-2-2b-it` и `meta-llama/Llama-3.2-3B-Instruct` на Hugging Face закрыты лицензией: нужно принять её на странице модели и выполнить `huggingface-cli login`.

**3. Бенчмарк.** Из папки `ml-cross-entropy`:

```bash
python -m benchmark --methods=cce,cce_kahan_full_c,cce_exact,torch_compile,baseline \
    --test_data=gemma2 --n_iteration=50 --n_rep=1 --output=../out/gemma2.csv

python -m benchmark --methods=cce,cce_kahan_full_c,cce_exact,torch_compile,baseline \
    --test_data=llama3.2-3 --n_iteration=50 --n_rep=1 --output=../out/llama3_2_3.csv
```

Другие модели из статьи: `gemma2-9`, `gemma2-27`, `phi3.5`, `mistral-nemo`, `llama3`. Чтобы получить ровно Table 1, поставьте `--n_rep=5`.

**4. Ноутбук.** [`reproduce.ipynb`](reproduce.ipynb) повторяет оба запуска, сводит CSV в таблицы и содержит эксперимент с LSTM. В ячейках остались абсолютные пути с сервера (`/disk/10tb/...`), их нужно заменить на свои.
