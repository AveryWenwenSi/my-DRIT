DATASET=$1

if [[ $DATASET != "summer2winter_yosemite" && $DATASET != "monet2photo" && $DATASET != "vangogh2photo" ]]; then
      echo "Available datasets are: apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos"
      exit 1
  fi

  echo "Specified [$DATASET]"
  URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/$DATASET.zip
  ZIP_FILE=../datasets/$DATASET.zip
  TARGET_DIR=../datasets/$DATASET/
  wget -N $URL -O $ZIP_FILE
  mkdir $TARGET_DIR
  unzip $ZIP_FILE -d ../datasets/
  rm $ZIP_FILE

if [[$DATASET != "portrait" && $DATASET != "cat2dog"]]; then
  exit
fi

URL=http://vllab.ucmerced.edu/hylee/DRIT/datasets/$DATASET.zip
wget -N $URL -O ../datasets/$DATASET.zip
unzip ../datasets/$DATASET.zip -d ../datasets
rm ../datasets/$DATASET.zip

