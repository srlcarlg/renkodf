## Changelog

### [2.0] - 2025-11-23

Well, when I wrote **[rangedf](https://github.com/srlcarlg/rangedf)** 6 months ago _(in order to use it in **[srl-python-indicators](https://github.com/srlcarlg/srl-python-indicators)**)_ using the same code struture from **renkodf (ver 1.1)**, <br>
I realized that the code **was really inefficient**, mostly because interacting 2x **_(1º on Tick data, 2º on Range Single Data)_** on dataframes was a waste of resources. <br>
So rewriting was necessary, **especially finding a way to calculate OHLC variations(modes) all at once.**

However, after writing srl-python-indicators, **some data analysis concepts are better developed when we're able to see it**, so I made the following roadmap: <br>
`srl-ctrader-indicators => renko/range(df) => srl-python-indicators`

Since I've some new requirements for my personal project (parallel backesting, low memory usage, ARM-compatible, etc...), <br>
extracting the **maximum performance of raw-implementation** before diving into restrictive approaches (like numba) is needed.

Key points of version 2.0:
- **Replace dict/list by numpy arrays**: Massive speed up of the loop through Tick Data!
    - I was not aware of the poor looping performance of python dictionaries until now:
    - https://www.geeksforgeeks.org/python/why-is-iterating-over-a-dictionary-slow-in-python/
- **OHLC modes**: All modes are calculated all at once!
    - When renko_df(mode) is called, it's just replaces the Renko OHLC columns by the respective mode OHL columns with vectorized operations, instead of looping through a dict and make a dataframe from it, as before.
- **Brick creation**: The conditions to create a new renko has been revised for better simplicity and speeeed.
    - The math for the Renko Price remains the same.

The **'test_...py'** files shares some logic of renkodf-java (from ver 1.1) tests, <br>
which I might update the java implementation with new v2.0 arrays logic soon.

The use of LLM was minimal, mostly for micro-optimization questions, the new **_[ create_renko() / renko_df() ]_** methods are all handcrafted, well-tested by a human being (myself, hahah).

### [1.1] - 2023-12-28
- **Great refactor** in `add_prices()` and `renko_df()` methods, improving logic, performance and readability. 
- **Added "show_progress"** in `Renko()` constructor, which was always enabled in the previous version, **negatively affecting performance by 2x.**
### [1.0] - 2023-07-11
First Release.
