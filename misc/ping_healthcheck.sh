#!/bin/bash

# Step 1: Check if the app is healthy
STATUS=$(curl -fs http://56.228.4.93:8000/healthz | grep -o '"status":"ok"')

# Step 2: If healthy, ping healthchecks.io
if [ "$STATUS" = '"status":"ok"' ]; then
  curl -fsS https://hc-ping.com/29110830-ba4f-4e8c-87a8-07646ab29f21 > /dev/null
fi