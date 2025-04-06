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
import pandas as pd

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

        # Modify type handling to be more robust
        if action.type:
            arg_type = action.type.__name__ if hasattr(action.type, '__name__') else str(action.type)
        else:
            arg_type = "str"

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

def display_directory_stats(dir_stats: dict):
    """Display directory statistics using Rich tables.
    
    Args:
        dir_stats (dict): Dictionary with structure {year: {groupname: {"lastUpdated": timestamp, "size": Megabytes}}}
    """
    console = Console()
    
    # Create main table
    main_table = Table(title="Directory Statistics", show_header=True, header_style="bold magenta")
    main_table.add_column("Year", style="cyan")
    main_table.add_column("Group Details")
    
    for year, groups in sorted(dir_stats.items()):
        # Create nested table for groups
        group_table = Table(show_header=True, header_style="bold blue", show_edge=False, padding=(0, 2))
        group_table.add_column("Group", style="green")
        group_table.add_column("Last Updated", style="yellow")
        group_table.add_column("Size (MB)", justify="right", style="red")
        
        for group_name, stats in sorted(groups.items()):
            group_table.add_row(
                group_name,
                stats["lastUpdated"].strftime("%Y-%m-%d %H:%M"),
                f"{stats['size']:.2f}"
            )
        
        main_table.add_row(str(year), group_table)
    
    console.print(main_table)
    
def create_table(dictionary, title):
    # Create a Rich console
    console = Console()

    # Create a table
    table = Table(title=title)

    # Add columns
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="magenta")

    # Add rows from the dictionary
    for key, value in dictionary.items():
        table.add_row(str(key), str(value))

    # Print the table
    console.print(table)

def print_dataframe_rich(df, title):
    console = Console()
    table = Table(
        title=f"[bold yellow]{title}[/bold yellow]",
        show_header=True,
        header_style="bold magenta",
        border_style="blue"
    )
    
    # Add index column first
    table.add_column("Index", style="bold cyan", justify="right")

    # Add remaining columns with different styles
    # Extended color palette for better distinction between columns
    colors = ["green", "magenta", "red", "blue", "yellow", "cyan", "purple",
             "bright_green", "bright_red", "bright_blue", "bright_magenta"]

    for idx, column in enumerate(df.columns):
        color = colors[idx % len(colors)]  # Cycle through colors if more columns than colors
        table.add_column(
            f"[bold {color}]{str(column)}[/bold {color}]",
            style=color,
            justify="center"
        )
    
    # Add rows without any dim styling
    for index, row in df.iterrows():
        row_values = [str(value) for value in row]
        table.add_row(str(index), *row_values)

    # Print the table with a surrounding panel
    panel = Panel.fit(
        table,
        border_style="bold blue",
        padding=(1, 2)
    )
    console.print(panel)

def visualize_csv(file_path, title=None, max_rows=None):
    """
    Visualize a CSV file using rich table formatting
    
    Args:
        file_path (str): Path to the CSV file
        title (str, optional): Title for the table. Defaults to filename
        max_rows (int, optional): Limit number of rows displayed
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Limit rows if specified
    if max_rows is not None:
        df = df.head(max_rows)
    
    # Use filename as default title if not provided
    if title is None:
        title = f"CSV: {file_path}"
    
    # Print the dataframe using rich table
    print_dataframe_rich(df, title)

def main():
    examples = [
        {
            "cmd": "python -m src.utils.displayutil data.csv",
            "desc": "Basic usage - display entire CSV"
        },
        {
            "cmd": "python -m src.utils.displayutil data.csv --title 'My Data' --max-rows 10",
            "desc": "Display first 10 rows with custom title"
        }
    ]
    
    parser = RichArgumentParser(
        description="CSV File Visualizer - Display CSV files in a rich, formatted table",
        examples=examples
    )
    
    parser.add_argument("file", help="Path to the CSV file to visualize")
    parser.add_argument("--title", help="Custom title for the table", default=None)
    parser.add_argument("--max-rows", type=int, help="Limit number of rows displayed", default=None)
    
    args = parser.parse_args()
    visualize_csv(args.file, args.title, args.max_rows)

if __name__ == "__main__":
    main()