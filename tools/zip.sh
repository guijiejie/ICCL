#!/bin/bash
zip -r myselfsup.zip bin configs modules imagenet_label runningscripts tools *.txt *.py *.sh *.json *.md -x "*.so" -x "runningscripts/*" -x "*/__pycache__/*" 
