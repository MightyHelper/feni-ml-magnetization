import rich.console
from rich.highlighter import ReprHighlighter
from rich.theme import Theme

col = "#333333"
console = rich.console.Console(theme=Theme({
	"zero.zero": col,
	"zero.zero_1": col,
	"zero.zero_2": col,
	"zero.zero_3": col,
}))
h = ReprHighlighter()
