# CoffeaMate -- An Analysis Toolkit for Coffea and Others

This repository contains utilities for data analysis packages like Coffea and others. It provides various tools and scripts to facilitate data processing, analysis, and visualization.

## Work In Progress
- Fixing minor memory leak issue
- Implementing flattened output

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