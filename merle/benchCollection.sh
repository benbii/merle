#!/bin/bash
set -e
shopt -s globstar
cd "$(dirname "$0")"
noRep=""
if [ "$1" = --noreplace ]; then
  shift
  noRep="--noreplace"
fi
resDir=${1:-res}
mkdir -p "$resDir"

profOne() {
  if ! [ -e "wahData/$1/199.wah" ]; then
    arrDir=$(echo teb/**/"$1")
    mkdir -p "wahData/$1"
    build/wahConv arrToWah --nFile 200 --nDup 100 \
      --inFmt "$arrDir/$1.csv%zu.txt" --outFmt "wahData/$1/%zu.wah"
  fi

  if [ -e "${4}/${1}_${2}.csv" ] && [ "$3" = --noreplace ]; then
    echo "$1 $2 exists"
  else
    echo "$1 $2 starts"
    if [ "${5:-bruh}" = --semi ]; then
      build/wahProfileGPU benchCollection --inFmt "wahData/$1/%zu.wah" \
        --stashOut "${4}/${1}_stash.csv" \
        --opOut "${4}/${1}_${2}.csv" --op "$2" --semi >/dev/null
    else
      build/wahProfileGPU benchCollection --inFmt "wahData/$1/%zu.wah" \
        --stashOut "${4}/${1}_stash.csv" \
        --opOut "${4}/${1}_${2}.csv" --op "$2" >/dev/null
    fi
  fi
}

profOne census-income_srt and "$noRep" "$resDir"
profOne census-income_srt or "$noRep" "$resDir"
profOne census-income and "$noRep" "$resDir" --semi
profOne census-income or "$noRep" "$resDir" --semi
profOne wikileaks-noquotes_srt and "$noRep" "$resDir"
profOne wikileaks-noquotes_srt or "$noRep" "$resDir"
profOne wikileaks-noquotes and "$noRep" "$resDir"
profOne wikileaks-noquotes or "$noRep" "$resDir"
profOne weather_sept_85_srt and "$noRep" "$resDir"
profOne weather_sept_85_srt or "$noRep" "$resDir"
profOne weather_sept_85 and "$noRep" "$resDir" --semi
profOne weather_sept_85 or "$noRep" "$resDir" --semi

profOne census-income_srt xor "$noRep" "$resDir"
profOne census-income xor "$noRep" "$resDir" --semi
profOne wikileaks-noquotes_srt xor "$noRep" "$resDir"
profOne wikileaks-noquotes xor "$noRep" "$resDir"
profOne weather_sept_85_srt xor "$noRep" "$resDir"
profOne weather_sept_85 xor "$noRep" "$resDir" --semi

profOne census-income_srt decode "$noRep" "$resDir"
profOne census-income decode "$noRep" "$resDir"
profOne wikileaks-noquotes_srt decode "$noRep" "$resDir"
profOne wikileaks-noquotes decode "$noRep" "$resDir"
profOne weather_sept_85_srt decode "$noRep" "$resDir"
profOne weather_sept_85 decode "$noRep" "$resDir"

# AFAIK the `rename` package in different distributions have various types of
# syntaxs so the following may likely fail. These commands rename produced
# results into those accepted in res/wahDrawFigs.ipynb
rename -i _srt Srt "$resDir/"*
rename -i weather_sept_85 wea "$resDir/"*
rename -i census-income inc "$resDir/"*
rename -i wikileaks-noquotes leak "$resDir/"*
