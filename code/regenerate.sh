#!/bin/bash

./cli dat rename-in-dataset ../dataset_versions/01_extra_shapes.csv --output-path ../dataset_versions/02_renamed.csv
./cli dat normalize-ratios ../dataset_versions/02_renamed.csv --output-path ../dataset_versions/03_normalized.csv
