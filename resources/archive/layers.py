
               


        # classification
        # self.prediction_layer = Dense(
        #     units = 1,
        #     activation = 'sigmoid'
        # )
        # probability = self.prediction_layer(logit)



            sparse_tensor = category_encoding(inputs[field])
            sparse_inputs = concat("sparse", sparse_inputs, sparse_tensor)
            dense_tensor = embedding(inputs[field])
            dense_inputs = concat("dense", dense_inputs, dense_tensor)




        elif name in metadata["cat_features"]:
            if name in cat_fields: # create categorical fields (used in the model)
                using_features.append(
                tf.feature_column.indicator_column(
                    tf.feature_column.categorical_column_with_vocabulary_list(
                        key = name, vocabulary_list = vocabs[name], num_oov_buckets = 0
                    )
                ))
            else: # add dummy Feature object just for the use of parsing
                dummy_features[name] = dummy_feature(field)
        else:
            dummy_features[name] = dummy_feature(field)