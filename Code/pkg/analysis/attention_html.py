"""
Render a per-token attention heat-map as a standalone HTML fragment.

Adapted from the `visualise()` function in the original Code/console.py, which wrote the
fragment to colorise.html on disk. Here it is returned as a string so the Streamlit app can
inject it directly and the CLI can still choose to write a file.
"""
from matplotlib.colors import rgb2hex
import matplotlib


def _reds():
    # matplotlib.cm.get_cmap was removed in matplotlib 3.9; matplotlib.colormaps is the
    # supported lookup and has been available since 3.5.
    return matplotlib.colormaps['Reds']


def render_attention_html(token_list: list, color_array, prediction=None) -> str:
    """
    Given attention weights and tokens, build a colour-mapped HTML fragment.
    :param token_list: list of tokens (strings)
    :param color_array: sequence of numbers between 0 and 1, parallel to token_list
    :param prediction: value between 0 and 1 to set the score bar to
    :return: an HTML string
    """
    cmap = _reds()
    template = ('<span style="color: black; font-size: 15px; line-height: 2.1; padding: 3px 1px; '
                'border-radius: 3px; background-color: {}">{}</span>')
    colored_string = '<div style="font-family: Arial, Helvetica, sans-serif;">'
    for t, color in zip(token_list, color_array):
        # if negative, set to white
        color_val = rgb2hex((1, 1, 1)) if color < 0 else rgb2hex(cmap(color)[:3])
        colored_string += template.format(color_val, '&nbsp;' + _escape(t) + '&nbsp;')

    colored_string += '</div>'

    if prediction is not None:
        colored_string += _render_score_bar(prediction)

    return colored_string


def _escape(text: str) -> str:
    """Tokens come from user input, so they must not be able to inject markup."""
    return (str(text).replace('&', '&amp;').replace('<', '&lt;')
            .replace('>', '&gt;').replace('"', '&quot;'))


def _render_score_bar(prediction: float) -> str:
    """A non-sarcastic -> sarcastic gradient bar with a marker at the prediction score."""
    percent = max(0.0, min(1.0, float(prediction))) * 100
    return (
        '<div style="margin-top: 18px; font-family: Arial, Helvetica, sans-serif;">'
        '<div style="position: relative; height: 12px; border-radius: 6px; '
        'background: linear-gradient(to right, #ff4c38, #ffff66, #85ff93);">'
        '<div style="position: absolute; left: {pct:.2f}%; top: -4px; width: 3px; height: 20px; '
        'background: #000; transform: translateX(-1.5px);"></div>'
        '</div>'
        '<div style="display: flex; justify-content: space-between; font-size: 11px; '
        'font-weight: bold; margin-top: 6px; opacity: 0.75;">'
        '<span>NON-SARCASTIC</span><span>NEUTRAL</span><span>SARCASTIC</span>'
        '</div></div>'
    ).format(pct=percent)


def write_attention_html(path: str, html: str) -> None:
    """Write a fragment to disk -- preserves the original colorise.html behaviour."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
