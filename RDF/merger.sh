SKOS="output_skos_categories_en"
SKOS_ALL="./${SKOS}_all.txt"
echo touch $SKOS_ALL

for i in $(seq -f "%02g" 0 5)
do
	echo "Merging ${SKOS}${i}"
	cat "${SKOS}${i}.txt" >> $SKOS_ALL
done

MAPPING="output_mappingbased_objects_en"
MAPPING_ALL="./${MAPPING}_all.txt"
echo touch $MAPPING_ALL

for i in $(seq -f "%02g" 0 12)
do
	echo "Merging ${MAPPING}${i}"
	cat "${MAPPING}${i}.txt" >> $MAPPING_ALL
done

INFOBOX="output_infobox_properties_mapped_en"
INFOBOX_ALL="./${INFOBOX}_all.txt"
echo touch $INFOBOX_ALL

for i in $(seq -f "%02g" 0 20)
do
	echo "Merging ${INFOBOX}${i}"
	cat "${INFOBOX}${i}.txt" >> $INFOBOX_ALL
done

ARTICLE="output_article_categories_en"
ARTICLE_ALL="./${ARTICLE}_all.txt"
echo touch $ARTICLE_ALL

for i in $(seq -f "%02g" 0 18)
do
	echo "Merging ${ARTICLE}${i}"
	cat "${ARTICLE}${i}.txt" >> $ARTICLE_ALL
done







