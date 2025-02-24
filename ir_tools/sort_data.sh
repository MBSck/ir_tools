#!/bin/bash

organise_matisse_data() {
  find $1 -type d -depth 1 | while read -r dir; do
    if [[ "$dir" != *"/non_treated" ]]; then
      if [[ "$dir" != *"/plots" ]]; then
        mkdir -p "$dir/plots"

        mv "$dir"/*.png "$dir/plots/" 2>/dev/null
      fi

      mkdir -p "$dir/non_treated"
      mv "$dir"/* "$dir/non_treated/" 2>/dev/null

      mkdir -p "$dir/matisse"
      mv "$dir/non_treated" "$dir/matisse"
    fi
  done
}
