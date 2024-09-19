#!/usr/bin/env bash

# MySQL credentials
DBUSER="mysqlbackup"
DBPASS="backup_password"
DATABASE="performance_schema"
TABLE="accounts"
BACKUPROOT="/mnt/d/data/mysqlbackup"

# Calculate today's date
DATEFORMAT=$(date +%F)
# Backup directory for today's date
BACKUPDIR="${BACKUPROOT}/${DATEFORMAT}"

# Ensure backup directory for today exists and is writable
if [ ! -d ${BACKUPDIR} ]; then
  echo "Attempting to create backup directory ${BACKUPDIR} ..."
  if ! mkdir -p ${BACKUPDIR}; then
    echo "Backup directory ${BACKUPDIR} could not be created by this user: ${USER}" 1>&2
    echo "Aborting..." 1>&2
    exit 1
  else
    echo "Directory ${BACKUPDIR} successfully created."
  fi
elif [ ! -w ${BACKUPDIR} ]; then
  echo "Backup directory ${BACKUPDIR} is not writable by this user: ${USER}" 1>&2
  echo "Aborting..." 1>&2
  exit 1
fi

# Export table data to CSV format using TCP/IP
mysql -h 127.0.0.1 -P 3306 -u $DBUSER -p$DBPASS -e "SELECT * FROM $DATABASE.$TABLE" | sed 's/\t/","/g;s/^/"/;s/$/"/;s/\n//g' > $BACKUPDIR/$DATEFORMAT-$DATABASE-$TABLE.csv

echo "Backup of table $TABLE in database $DATABASE completed for today's date ($DATEFORMAT) in CSV format."
