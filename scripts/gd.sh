#!/bin/bash
if [ $# -eq 2 ];then
    FILEID=$1
    FILE_PATH=$2

    echo "Downloading from URL $FILEID into $FILE_PATH"

    FILEID="$(echo $FILEID | sed -n 's#.*\https\:\/\/drive\.google\.com/file/d/\([^.]*\)\/view.*#\1#;p')";
    FILENAME="$(wget -q --show-progress -O - "https://drive.google.com/file/d/$FILEID/view" | sed -n -e 's!.*<title>\(.*\)\ \-\ Google\ Drive</title>.*!\1!p')";
    wget -q --show-progress --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget -q --show-progress --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$FILEID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILEID" -c -O "$FILE_PATH" && rm -rf /tmp/cookies.txt;
    echo "file $FILENAME has been downloaded in $FILE_PATH"

else
    echo "insert the google drive full url to download";
    exit 1
fi
