#!/bin/bash
# Download the TRENDY S2 (and S3) model output used here (nbp, landCoverFrac, fFire, fLuc, cVeg) from the TRENDY
# server with lftp. Access requires a TRENDY account. Usage: download_trendy.sh USER REMOTE_DIR LOCAL_DIR
# (TRENDY v13 / GCB2024: REMOTE_DIR=/gcb2024/LAND/OUTPUT)
SERVER="trendydl.exeter.ac.uk"
USER="${1:?Usage: download_trendy.sh USER REMOTE_DIR LOCAL_DIR}"
REMOTE_DIR="${2:?REMOTE_DIR}"
LOCAL_DIR="${3:?LOCAL_DIR}"

read -s -p "Enter password for $USER@$SERVER: " PASSWORD
echo ""

lftp -u $USER,$PASSWORD sftp://$SERVER <<EOF2
cd $REMOTE_DIR

mirror -vvv --verbose --target-directory="$LOCAL_DIR" -I '*S2*nbp.*' -I '*S2*landCoverFrac.*' -I '*S2*oceanCoverFrac.*' -I '*S2*fLuc.*' -I '*S2*fFire.*' -I '*S2*cVeg.*'
mirror -vvv --verbose --target-directory="$LOCAL_DIR" -I '*S3*nbp.*' -I '*S3*landCoverFrac.*' -I '*S3*oceanCoverFrac.*' -I '*S3*fLuc.*' -I '*S3*fFire.*' -I '*S3*cVeg.*'

bye
EOF2
