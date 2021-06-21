SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
DATA_DIR=${SCRIPTPATH}/data

echo "Downloading data in ${DATA_DIR}"
if [ -d $DATA_DIR ]
then
    echo "$DATA_DIR already exists"
else
    echo "Creating folder $DATA_DIR"
    mkdir $DATA_DIR
fi

wget https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip --no-check-certificate
unzip shapenetcore_partanno_segmentation_benchmark_v0.zip -d $DATA_DIR
rm shapenetcore_partanno_segmentation_benchmark_v0.zip
