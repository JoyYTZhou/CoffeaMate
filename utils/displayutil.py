from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.logging import RichHandler
from rich.traceback import install
from rich.progress import track
from rich.markdown import Markdown
import argparse
from typing import List, Dict, Optional, Union

class RichArgumentParser(argparse.ArgumentParser):
    """Custom argument parser that uses rich for help display"""
    def __init__(self, 
                 *args, 
                 examples: Optional[List[Dict[str, str]]] = None,
                 custom_usage: Optional[str] = None,
                 **kwargs):
        """
        Initialize the Rich Argument Parser.
        
        Args:
            examples: List of dictionaries containing example commands and their descriptions
                     [{'cmd': 'python script.py arg1', 'desc': 'Basic usage'}]
            custom_usage: Custom usage markdown string to display instead of examples
            *args, **kwargs: Standard argparse.ArgumentParser arguments
        """
        super().__init__(*args, **kwargs)
        self.console = Console()
        self.examples = examples
        self.custom_usage = custom_usage

    def print_help(self):
        """Override the print_help method to use rich formatting"""
        self._print_title()
        self._print_arguments_table()
        self._print_usage()

    def _print_title(self):
        """Print the title panel"""
        title = Panel.fit(
            f"[bold yellow]{self.description}[/bold yellow]",
            border_style="blue",
            padding=(1, 2)
        )
        self.console.print(title)

    def _print_arguments_table(self):
        """Print the arguments table"""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Argument", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Description", style="white")
        table.add_column("Default", style="yellow")

        # Add positional arguments
        for action in self._actions:
            if not action.option_strings:
                self._add_argument_to_table(table, action, is_positional=True)

        # Add optional arguments
        for action in self._actions:
            if action.option_strings:
                self._add_argument_to_table(table, action, is_positional=False)

        self.console.print(table)

    def _add_argument_to_table(self, table: Table, action: argparse.Action, is_positional: bool):
        """Add an argument to the table"""
        if is_positional:
            name = action.dest
            default = "[red]required[/red]"
        else:
            name = ", ".join(action.option_strings)
            default = str(action.default) if action.default is not None else "None"

        arg_type = str(action.type).__name__ if action.type else "str"
        help_text = action.help or ""
        if action.choices:
            help_text += f" Choices: {action.choices}"

        table.add_row(name, arg_type, help_text, default)

    def _print_usage(self):
        """Print usage examples or custom usage"""
        if self.custom_usage:
            self.console.print(Markdown(self.custom_usage))
        elif self.examples:
            examples_md = "### Examples:\n\n"
            for example in self.examples:
                examples_md += f"**{example.get('desc', 'Example')}:**\n"
                examples_md += f"```bash\n{example['cmd']}\n```\n\n"
            self.console.print(Markdown(examples_md))

def create_rich_logger() -> RichHandler:
    """Create a Rich logger handler with standard configuration"""
    return RichHandler(
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        show_time=True
    )

def create_status_console() -> Console:
    """Create a console for status updates"""
    return Console()

if __name__ == "__main__":
    # Example of how to use the RichArgumentParser
    examples = [
        {
            "cmd": "python script.py input_file --output result.txt",
            "desc": "Basic usage"
        },
        {
            "cmd": "python script.py input_file --verbose --format json",
            "desc": "Advanced usage with options"
        }
    ]
    
    parser = RichArgumentParser(
        description="Example Script",
        examples=examples
    )
    parser.add_argument("input_file", help="Input file to process")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--verbose", action="store_true", help="Increase output verbosity")
    
    parser.print_help()