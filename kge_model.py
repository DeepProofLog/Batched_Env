from keras import Model, regularizers, models
from keras.layers import Dense, Dropout, Lambda, Layer
import ns_lib as ns
import tensorflow as tf
from ns_lib.nn.constant_embedding import *
from ns_lib.nn.reasoning import *
from ns_lib.nn.kge import KGEFactory, KGELayer
from ns_lib.logic import FOL, Rule
from ns_lib.logic.semantics import GodelTNorm
from typing import Dict, List, Union


class KGEModel(Model):

    def __init__(self, fol:FOL,
                 kge: str,
                 kge_regularization: float,
                 constant_embedding_size: int,
                 predicate_embedding_size: int,
                 kge_atom_embedding_size: int,
                 kge_dropout_rate: float,
                 num_adaptive_constants: int=0,
                 dot_product: bool=False,
                 trainable: bool=False,
                 dtype=tf.int32):
        super().__init__()
        self.fol = fol
        self.predicate_index_tensor = tf.constant(
            [i for i in range(len(self.fol.predicates))], dtype=tf.int32)
                
        # CONSTANT AND PREDICATE EMBEDDINGS
        self.predicate_embedder = PredicateEmbeddings(
            fol.predicates,
            predicate_embedding_size,
            regularization=kge_regularization,
            has_features=False)
        self.constant_embedder = ConstantEmbeddings(
            domains=fol.domains,
            constant_embedding_sizes_per_domain={
                domain.name: constant_embedding_size
                for domain in fol.domains},
            regularization=kge_regularization,
            has_features=False)
        self.dot_product = dot_product
        
        if num_adaptive_constants > 0:
            self.adaptive_constant_embedder = AdaptiveConstantEmbeddings(
                domains=fol.domains,
                constant_embedder=self.constant_embedder,
                constant_embedding_size=constant_embedding_size,
                num_adaptive_constants=num_adaptive_constants,
                dot_product=dot_product)
        else:
            self.adaptive_constant_embedder = None

        # ATOM EMBEDDINGS
        self.kge_embedder, self.output_layer = KGEFactory(
            name=kge,
            atom_embedding_size=kge_atom_embedding_size,
            relation_embedding_size=kge_atom_embedding_size,
            regularization=kge_regularization,
            dropout_rate=kge_dropout_rate)
        assert self.kge_embedder is not None

    def create_triplets(self,
                        constant_embeddings: Dict[str, tf.Tensor],
                        predicate_embeddings: tf.Tensor,
                        A_predicates: Dict[str, tf.Tensor]):
        
        predicate_embeddings_per_triplets = []
        '''For A_predicates, take the emebdding representation of the predicates and the constants and create the triplets for the KGE model.
        For instance, if I have a predicate with 3 grounded atoms, I will repeat the embedding of that predicate 3 times, and put it with the 
        embeddings of the constants for each grounded atom
        - output: 
            predicate_embeddings_per_triplets: [n_predicates, n_atoms/grounding per predicate, embed_size_predicate]
            constant_embeddings_for_triplets: [n_atoms,2=n_domains,embed_size_constant]'''

        for p,indices in A_predicates.items():
            idx = self.fol.name2predicate_idx[p]
            # Repeat the predicate embedding for each atom in the predicate
            p_embeddings = tf.expand_dims(predicate_embeddings[idx], axis=0)  # [1,1,200]=[1,1, embed_size_predicate]
            predicate_embeddings_per_triplets.append(tf.repeat(p_embeddings, tf.shape(indices)[0], axis=0))  # [1,1918,200]=[1,n_groundings for that predicate, embed_size_predicate]
        predicate_embeddings_per_triplets = tf.concat(predicate_embeddings_per_triplets,axis=0) # shape=[n_predicates, n_groundings for that predicate, embed_size_predicate]
        constant_embeddings_for_triplets = []

        for p,constant_idx in A_predicates.items():
            constant_idx = tf.cast(constant_idx, tf.int32) # all the groundings idx for that predicate
            predicate = self.fol.name2predicate[p]
            one_predicate_constant_embeddings = []
            '''Here, for each domain of the predicate (LocInSR(subregion,region)->for subregion), I get the embeddings of the constants that
            are grounded in that domain. If LocInSR has 58 groundings/atoms, I will get the representation of the subregion constants in 
            those atoms(58,200). I do the same for the domain region, so I get a tensor of shape [58,2,200] for LocIn where 2 is the arity of the predicate.''' 
            for i,domain in enumerate(predicate.domains):
                '''If I have A_pred=[country,region]=[[1,2],[3,4],...], I get for country: [1,3] which are local! they're the pos of the ctes in X_domain
                In X_domain, in pos i I have the global idx of that cte (which has been created to create the embedds)'''
                constants = tf.gather(constant_embeddings[domain.name],
                                    constant_idx[..., i], axis=0) # constant_idx[..., i] takes the idx of the constants for that domain (in predicate p)
                one_predicate_constant_embeddings.append(constants)
                    
            # shape (predicate_batch_size, predicate_arity, constant_embedding_size)
            one_predicate_constant_embeddings = tf.stack(one_predicate_constant_embeddings,axis=-2)
            constant_embeddings_for_triplets.append(one_predicate_constant_embeddings)
            
        '''For all the queries, I have divided them by predicates. Once I have, for each predicate, the embeddings of the constants, i.e., 
        for LocInSR I have 58 atoms -> (58,2,200), for NeighOf .... I concatenate them to get a tensor of shape [58+..,2,200] = [3889,2,200]
        tf.print('PREDICATE EMBEDDINGS PER TRIPLETS',predicate_embeddings_per_triplets.shape,predicate_embeddings_per_triplets)
        tf.print('CONSTANT EMBEDDINGS FOR TRIPLETS', [tensor.shape for tensor in constant_embeddings_for_triplets])'''

        constant_embeddings_for_triplets = tf.concat(constant_embeddings_for_triplets,axis=0) 
        tf.debugging.assert_equal(tf.shape(predicate_embeddings_per_triplets)[0],
                                  tf.shape(constant_embeddings_for_triplets)[0])
        # Shape TE, T2E with T number of triplets.
        # At the end I get for both predicates and embeddings a tensor of shape [n_atoms,ctes_in_atoms(arity),embed_size_predicate] and [n_atoms,2,embed_size_constant]
        return predicate_embeddings_per_triplets, constant_embeddings_for_triplets

    def call(self, inputs):
        '''
        X_domains type is Dict[str, inputs]
        A_predicate: Dict[predicate_name, List[Tuple[Index1, ..., IndexN]]]
        '''
        (X_domains, A_predicates) = inputs
        if self.adaptive_constant_embedder is not None:
            # Create a mask to fix the values that are not in the domain.
            X_domains_fixed_mask = {
                name:tf.where(x < len(self.fol.name2domain[name].constants),
                              True, False) for name,x in X_domains.items()}
            # Set to 0 the values that are not in the domain.
            X_domains_fixed = {
                name:tf.where(X_domains_fixed_mask[name], x, 0)
                for name,x in X_domains.items()}
            # Get the embeddings for the fixed values and the adaptive values.
            constant_embeddings_fixed = self.constant_embedder(X_domains_fixed)
            constant_embeddings_adaptive = self.adaptive_constant_embedder(
                X_domains)
            constant_embeddings = {
                name:tf.where(
                    # Expand dim to broadcast to the embeddings size.
                    tf.expand_dims(X_domains_fixed_mask[name], axis=-1),
                    constant_embeddings_fixed[name],
                    constant_embeddings_adaptive[name])
                for name in X_domains.keys()}
        else:
            constant_embeddings = self.constant_embedder(X_domains) # For the constant embedds, I always need global idx to get consistent embedds every batch
        predicate_embeddings = self.predicate_embedder(self.predicate_index_tensor) # Embedds for every pred in fol (global idx)

        # Given the embedds of the constants and the predicates, I create the triplets with the embeddings of the atoms and the predicates. 
        # A_predicates indicates the indeces of the constants for each grounding of the predicate, i.e., the queries
        predicate_embeddings_per_triplets, constant_embeddings_for_triplets = \
            self.create_triplets(constant_embeddings, predicate_embeddings, A_predicates)
        # Given the triplets with their embeddings obtained in create_triplets, I get the embeddings of the atoms with e.g. Transe
        atom_embeddings = self.kge_embedder((predicate_embeddings_per_triplets,
                                             constant_embeddings_for_triplets))
        # Get the score for each atom
        atom_outputs = tf.expand_dims(self.output_layer(atom_embeddings), -1)
        return atom_outputs, atom_embeddings


class CollectiveModel(Model):

    def __init__(self,
                 fol: FOL,
                 rules: List[Rule],
                 *,  # all named after this point
                 kge: str,
                 kge_regularization: float,
                 constant_embedding_size: int,
                 predicate_embedding_size: int,
                 kge_atom_embedding_size: int,
                 kge_dropout_rate: float,
                 reasoner_atom_embedding_size: int,
                 reasoner_formula_hidden_embedding_size: int,
                 reasoner_regularization: float,
                 reasoner_single_model: bool,
                 reasoner_dropout_rate: float,
                 reasoner_depth: int,
                 aggregation_type: str,
                 signed: bool,
                 temperature: float,
                 model_name: str,
                 resnet: bool,
                 embedding_resnet: bool,
                 filter_num_heads: int,
                 filter_activity_regularization: float,
                 num_adaptive_constants: int,
                 dot_product: bool,
                 cdcr_use_positional_embeddings: bool,
                 cdcr_num_formulas: int,
                 r2n_prediction_type: str,
                 distill: bool):
        super().__init__()
        self.distill = distill
        # Reasoning depth of the model structure.
        self.reasoner_depth = reasoner_depth
        # Reasoning depth currently used, this can differ from  self.reasoner_depth during multi-stage learning (like if pretraining the KGEs).
        self.enabled_reasoner_depth = reasoner_depth
        self.resnet = resnet
        self.embedding_resnet = embedding_resnet
        self.logic = GodelTNorm()

        self.kge_model = KGEModel(fol, kge,
                                  kge_regularization,
                                  constant_embedding_size,
                                  predicate_embedding_size,
                                  kge_atom_embedding_size,
                                  kge_dropout_rate,
                                  num_adaptive_constants,
                                  dot_product)
        self.model_name = model_name

        # REASONING LAYER
        self.reasoning = None
        if reasoner_depth > 0 and len(rules) > 0:
            if self.embedding_resnet:
                self.embedding_resnet_weight = Sequential([
                    #Dense(16, activation='relu'),
                    #Dropout(0.2),
                    Dense(1,
                          #kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                          bias_regularizer=regularizers.L2(1e-4),
                          activity_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-5),
                          activation='sigmoid')])

            self.reasoning = []
            for i in range(reasoner_depth):
              if i > 0 and reasoner_single_model:
                  self.reasoning.append(self.reasoning[0])
                  continue

              if model_name == 'dcr':
                  self.reasoning.append(DCRReasoningLayer(
                      templates=rules,
                      formula_hidden_size=reasoner_formula_hidden_embedding_size,
                      aggregation_type=aggregation_type,
                      temperature=temperature,
                      signed=signed,
                      filter_num_heads=filter_num_heads,
                      regularization=reasoner_regularization,
                      dropout_rate=reasoner_dropout_rate))

              elif model_name == 'cdcr':
                  self.reasoning.append(ClusteredDCRReasoningLayer(
                      num_formulas=cdcr_num_formulas,
                      use_positional_embeddings=cdcr_use_positional_embeddings,
                      templates=rules,
                      formula_hidden_size=reasoner_formula_hidden_embedding_size,
                      aggregation_type=aggregation_type,
                      temperature=temperature,
                      signed=signed,
                      filter_num_heads=filter_num_heads,
                      regularization=reasoner_regularization,
                      dropout_rate=reasoner_dropout_rate))

              elif model_name == 'r2n':
                  def SumAndSigmoidOutput(x):
                      return tf.nn.sigmoid(tf.reduce_sum(x, axis=-1))
                  output_layer = (Lambda(SumAndSigmoidOutput, name='output_layer')
                                  if kge == 'rotate' else self.kge_model.output_layer)
                  self.reasoning.append(R2NReasoningLayer(
                      rules=rules,
                      formula_hidden_size=reasoner_formula_hidden_embedding_size,
                      atom_embedding_size=reasoner_atom_embedding_size,
                      prediction_type=r2n_prediction_type,
                      aggregation_type=aggregation_type,
                      output_layer=output_layer,
                      regularization=reasoner_regularization,
                      dropout_rate=reasoner_dropout_rate))

              elif model_name == 'sbr':
                  self.reasoning.append(SBRReasoningLayer(
                      rules=rules,
                      aggregation_type=aggregation_type))

              elif model_name == 'gsbr':
                  self.reasoning.append(GatedSBRReasoningLayer(
                      rules=rules,
                      aggregation_type=aggregation_type,
                      regularization=reasoner_regularization))

              elif model_name == 'rnm':
                  self.reasoning.append(RNMReasoningLayer(
                      rules=rules,
                      aggregation_type=aggregation_type,
                      regularization=reasoner_regularization))

              elif model_name == 'dsl':
                  self.reasoning.append(DeepStocklogLayer(
                      rules=rules,
                      aggregation_type=aggregation_type,
                      regularization=reasoner_regularization))

              else:
                  assert False, 'Unknown model name %s' % model_name

        self._explain_mode = False

    def explain_mode(self, mode=True):
        self._explain_mode = mode

    @tf.function(reduce_retracing=True)
    def call(self, inputs, training=False, *args, **kwargs):
        '''
        X_domains type is Dict[str, tensor[constant_indices_in_domain]]
        A_predicate: Dict[predicate_name, List[Tuple[Index1, ..., IndexN]]]
                     e.g. mapping predicate_name -> tensor [num_groundings, arity]
                     with constant indices for each grounding.
        '''

        if self._explain_mode:
            # No explanations are posible when reasoning is disabled.
            assert self.reasoning is not None
            # Check that we are using an explainable model.
            assert self.model_name == 'dcr' or self.model_name == 'cdcr'

        # X_domains type is Dict[str, tensor[constant_indices_in_domain]]
        # A_predicate: Dict[predicate_name, List[Tuple[Index1, ..., IndexN]]]
        #              e.g. mapping predicate_name -> tensor [num_groundings, arity]
        #                   with constant indices for each grounding.
        (X_domains, A_predicates, A_rules, Q) = inputs
        concept_output, concept_embeddings = self.kge_model((X_domains, A_predicates))
        task_output = tf.identity(concept_output) # (len(sum A_pred),1)

        explanations = None
        if self.reasoning is not None:
            atom_embeddings = tf.identity(concept_embeddings)
            for i in range(self.enabled_reasoner_depth):
                if self._explain_mode and i == self.enabled_reasoner_depth - 1:
                    explanations = self.reasoning[i].explain(
                        [task_output, atom_embeddings, A_rules])
                task_output, atom_embeddings = self.reasoning[i]([
                    task_output, atom_embeddings, A_rules])
            if self.embedding_resnet:
                # In this case we need to recompute the output from the updated embeddings.
                w = tf.clip_by_value(self.embedding_resnet_weight(tf.concat([concept_embeddings, atom_embeddings], axis=-1)), 1e-9, 1.0 - 1e-7)
                tf.print('embedding_resnet_weight', tf.reduce_mean(w))
                atom_embeddings = (1.0 - w) * tf.stop_gradient(concept_embeddings) + w * atom_embeddings
                task_output = tf.expand_dims(self.output_layer(atom_embeddings), axis=-1)

        task_output = tf.gather(params=tf.squeeze(task_output, -1), indices=Q)
        concept_output = tf.gather(params=tf.squeeze(concept_output, -1),indices=Q)

        def safe_to_tensor(x):
            return x.to_tensor() if isinstance(x, tf.RaggedTensor) else x

        # Always work with the dense version
        concept_output_dense = safe_to_tensor(concept_output)
        task_output_dense = safe_to_tensor(task_output)

        # Ensure self.distill is a tensor (not a Python bool)
        distill_cond = tf.cast(self.distill, tf.bool)

        # Use tf.cond so that when distill is False, the outputs are zeros
        concept_output_ = tf.cond(
            distill_cond,
            lambda: concept_output_dense,
            lambda: tf.zeros_like(concept_output_dense)
        )

        task_output_ = tf.cond(
            distill_cond,
            lambda: task_output_dense,
            lambda: tf.zeros_like(task_output_dense)
        )

        non_empty = tf.logical_and(
            tf.reduce_any(tf.shape(concept_output_) > 0),
            tf.reduce_any(tf.shape(task_output_) > 0)
        )

        diff = concept_output_ - task_output_
        loss = tf.where(
            non_empty,
            diff * diff,
            tf.constant(0.0)
        )

        self.add_loss(loss)

        if self.resnet and self.reasoning is not None:
            task_output = self.logic.disj_pair(task_output, concept_output)

        if self._explain_mode:
            return concept_output, task_output, explanations
        else:
            return {'concept':concept_output, 'task':task_output}