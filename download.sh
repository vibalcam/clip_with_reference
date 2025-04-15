lines=(
"Processing file 1kEa1SWnT0AYnMKAzSBqGA2d0vXpRLovv part00.tar"
"Processing file 1fHLSaaRO_AHFvMTbTeQ6J742U0fkSW25 part01.tar"
"Processing file 1YS_sQMeFwYMs7oR1mPwNDqPR8hUegSmf part02.tar"
"Processing file 1fL0SK5o__5E2XbAIQEZtLsVkZWQQnwbW part03.tar"
"Processing file 1_9HNZQSoifAi3qVyLVt22X212BfBVeJ0 part04.tar"
"Processing file 1fEnb9khgINjnAZIsP6oMD9jlRNYI4D67 part05.tar"
"Processing file 142OXwrCJcQrOAzvi9nferPCqjDRARR5N part06.tar"
)

# https://stackoverflow.com/a/67550427
if [ -z "$TOKEN" ]; then
    echo "The TOKEN environment variable must be set"
    exit 1
fi

untar() {
    local src="$1"
    local target="$2"
    echo "$src >> $target"
    curl -H "Authorization: Bearer ${TOKEN}" https://www.googleapis.com/drive/v3/files/${src}?alt=media -o $target
    tar xf $target
    rm $target
}

mkdir datasets
cd ./datasets/dfn_data

for i in "${!lines[@]}"; do
    line=${lines[$i]}
    src=$(echo $line | awk '{print $3}')
    target=$(echo $line | awk '{print $4}')
    untar "$src" "$target"  & 
done


