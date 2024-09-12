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
    - [`Processor`](#processor)
    - [`BaseEventSelections` Class](#baseeventselections-class)
    - [`Object` Class](#object-class)
    - [`weightedCutflow` Class](#weightedcutflow-class)
    - [`weightedCutflow` Class](#weightedcutflow-class-1)
  - [Adding as a submodule](#adding-as-a-submodule)

## Features
- Utilities for Coffea and other data analysis packages
- Data processing scripts
- Analysis and visualization tools

## Useful Classes Overview

### `Processor`

The `Processor` class is designed to handle the processing of datasets, including initialization, directory setup, and remote file loading. Below is a brief overview of its key components:

### `BaseEventSelections` Class

The `BaseEventSelections` class serves as a base class for event selections. It provides a framework for applying various selection criteria to events based on trigger, object, and mapping configurations.

### `Object` Class

The `Object` class is designed for handling object selections and serves as an observer of the events. It provides a framework for managing selection and mapping configurations for different objects such as Electrons, Muons, and Jets.

### `weightedCutflow` Class

The `weightedCutflow` class extends the `Cutflow` class and is designed to handle weighted cutflows. It provides methods for initializing, adding, and retrieving the results of the cutflow.

### `weightedCutflow` Class

The `weightedCutflow` class extends the `Cutflow` class and is designed to handle weighted cutflows. It provides methods for initializing, adding, and retrieving the results of the cutflow.

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