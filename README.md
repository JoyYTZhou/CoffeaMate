# CoffeaMate -- An Analysis Toolkit for Coffea and Others

This repository contains utilities for data analysis packages like Coffea and others. It provides various tools and scripts to facilitate data processing, analysis, and visualization.

## Work In Progress
- Fixing minor memory leak issue
- Implementing flattened output

## Table of Contents
- [CoffeaMate -- An Analysis Toolkit for Coffea and Others](#coffeamate----an-analysis-toolkit-for-coffea-and-others)
  - [Work In Progress](#work-in-progress)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Useful Classes Overview](#useful-classes-overview)
    - [`JobLoader` Class](#jobloader-class)
    - [`JobRunner` Class](#jobrunner-class)
    - [`Processor`](#processor)
    - [`BaseEventSelections` Class](#baseeventselections-class)
    - [`Object` Class](#object-class)
    - [`weightedCutflow` Class](#weightedcutflow-class)
    - [`weightedSelection` Class](#weightedselection-class)
    - [`CSVPlotter` Class](#csvplotter-class)
  - [Adding as a submodule](#adding-as-a-submodule)

## Features
- Utilities for Coffea and other data analysis packages
- Data processing scripts
- Analysis and visualization tools

## Useful Classes Overview

### `JobLoader` Class

The `JobLoader` class is responsible for loading meta job files and preparing them for processing by slicing the files into smaller jobs. It handles the initialization of paths, checking for existing paths, and writing job parameters to JSON files.

### `JobRunner` Class

The `JobRunner` class is responsible for initializing and managing the submission of jobs based on the provided run settings, job file, and event selection class. It supports both synchronous and asynchronous (WIP) job submission using a Dask client.

### `Processor`

The `Processor` class is designed to handle the processing of datasets, including initialization, directory setup, and remote file loading/transferring.

### `BaseEventSelections` Class

The `BaseEventSelections` class serves as a base class for event selections. It provides a framework for applying various selection criteria to events based on trigger, object, and mapping configurations.

### `Object` Class

The `Object` class is designed for handling object selections and serves as an observer of the events. It provides a framework for managing selection and mapping configurations for different objects such as Electrons, Muons, and Jets.

### `weightedCutflow` Class

The `weightedCutflow` class extends the `Cutflow` class and is designed to handle weighted cutflows. It provides methods for initializing, adding, and retrieving the results of the cutflow.

### `weightedSelection` Class

The `weightedSelection` class extends the `PackedSelection` class and represents a set of selections on a set of events with weights. It provides methods for adding selections sequentially and generating a weighted cutflow.

### `CSVPlotter` Class

The `CSVPlotter` class is designed to plot histograms and other visualizations from CSV files. It utilizes various libraries such as `mplhep`, `matplotlib`, `numpy`, and `pandas` to create and manage plots. The class is initialized with an output directory and a configuration object for plotting.

## Adding as a submodule

To add this repository as a submodule to your existing Git repository, follow these steps:

1. Navigate to the root directory of your Git repository:
    ```bash
    cd /path/to/your/repo
    ```

2. Add this repository as a submodule:
    ```bash
    git submodule add <repository-url> path/to/submodule
    ```

3. Initialize and update the submodule:
    ```bash
    git submodule update --init --recursive
    ```

4. Commit the changes:
    ```bash
    git add .gitmodules path/to/submodule
    git commit -m "Add submodule for Coffea utilities"
    ```

To update the submodule to the latest version, navigate to the submodule directory and pull the latest changes:
```bash
cd path/to/submodule
git pull origin main