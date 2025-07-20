
CARS196_ROOT='data/cars196/'
CARS196_DATA='1xWSUMQIME3yqK8_NAdoBzYu8kO8bk4gq'


if [[ ! -d "${CARS196_ROOT}" ]]; then
    mkdir -p data/cars196/
    pushd data/cars196/
    echo "Downloading Cars196 data-set..."
    gdown "${CARS196_DATA}" -O cars196.tar
    tar -xvf cars196.tar
    rm cars196.tar
    popd
fi


CUB_ROOT='data/CUB_200_2011/'
CUB_DATA='https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz'


if [[ ! -d "${CUB_ROOT}" ]]; then
    mkdir -p data/
    pushd data/
    echo "Downloading CUB_200_2011 data-set..."
    wget ${CUB_DATA}
    tar -zxf CUB_200_2011.tgz
    rm CUB_200_2011.tgz
    popd
fi





SOP_ROOT='data/Stanford_Online_Products/'
SOP_DATA='https://www.dropbox.com/scl/fi/7icj466ds04ex7rd7kxxs/online_products.tar?rlkey=c2tp644h3uzui38tpu3l8z2uq&e=1&dl=0'


if [[ ! -d "${SOP_ROOT}" ]]; then
    mkdir -p data/
    pushd data/
    echo "Downloading Stanford Online Products dataset..."
    wget -O online_products.tar "${SOP_DATA}"
    tar -xvf online_products.tar
    rm online_products.tar
    mv online_products Stanford_Online_Products
    popd
fi





INSHOP_ROOT='data/Inshop_Clothes/'
INSHOP_DATA='0B7EVK8r0v71pVDZFQXRsMDZCX1Ek'


if [[ ! -d "${INSHOP_ROOT}" ]]; then
    mkdir -p data/
    pushd data/
    echo "Downloading INSHOP dataset..."
    gdown "${INSHOP_DATA}" -O Inshop_Clothes.zip
    unzip Inshop_Clothes.zip
    rm Inshop_Clothes.zip
    popd
fi






