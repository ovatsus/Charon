﻿namespace Charon

module DecisionTree =

    open Charon
    open System
    open System.Collections.Generic

    type To = 
        | String of string
        | Int of int
        | Int64 of int64
        | Bool of bool
        | Decimal of decimal
        | Float of float
        static member Feature(x) = x |> Option.map String
        static member Feature(x) = x |> Option.map Int
        static member Feature(x) = x |> Option.map Int64
        static member Feature(x) = x |> Option.map Bool
        static member Feature(x) = x |> Option.map Decimal
        static member Feature(x) = x |> Option.map Float

    // A feature maps the outcomes, encoded as integers,
    // to sorted observation indexes in the dataset.
    type Feature = Map<int, index>
    // Training Set = Labels and Features
    type TrainingSet = Feature * Feature []

    // A tree is either 
    // a Leaf (a final conclusion has been reached), or
    // a Branch (depending on the outcome on a feature,
    // more investigation is needed).
    type Tree = 
    | Leaf of int // decision
    | Branch of int * int * Map<int, Tree> // feature index, default choice, & sub-trees by outcome

    let private h (category: int) (total: int) = 
        if total = 0 then 0. // is this right? failwith "At least one observation is needed."
        elif category = 0 then 0.
        else
            let p = (float)category / (float)total
            - p * log p

    // Total elements in a feature.
    let private total (f: Feature) = 
        f 
        |> Map.toSeq 
        |> Seq.sumBy (fun (x, y) -> Index.length y)
    
    // Entropy of a feature.
    let entropy (f: Feature) =
        let size = total f
        f 
        |> Map.toSeq 
        |> Seq.sumBy (fun (x,y) -> h (Index.length y) size)

    // Apply a filter to a Feature, retaining
    // only the elements whose index are in the filter set.
    let filterBy (filter: int list) (feature: Feature) =
        feature
        |> Map.map (fun value indexes -> Index.intersect indexes filter)

    // Retrieve all indexes covered by feature
    let indexesOf (feature: Feature) =
        feature 
        |> Map.fold (fun indexes k kIndexes -> Index.merge indexes kIndexes) []

    // Split labels based on the values of a feature
    let split (feature: Feature) (labels: Feature) =
        feature 
        |> Map.map (fun v indexes ->
               labels 
               |> Map.map (fun l labelIndexes -> 
                      Index.intersect labelIndexes indexes))

    // Conditional Entropy when splitting labels by feature
    let conditionalEntropy (feature: Feature) (labels: Feature) =
        let size = total feature
        split feature labels
        |> Map.map (fun v feature -> 
               (float)(total feature) * entropy (labels |> filterBy (indexesOf feature)) / (float)size)
        |> Map.toSeq
        |> Seq.sumBy snd
    
    // Given a filter on indexes and remaining features,
    // pick the feature that yields highest information gain
    // to split the tree on.             
    let selectFeature (dataset: TrainingSet) // full dataset
                      (filter: index) // indexes of observations in use
                      (remaining: int Set) = // indexes of features usable 
        let labels = fst dataset |> filterBy filter
        let initialEntropy = entropy labels
        
        let best =            
            remaining
            |> Seq.map (fun f -> f, (snd dataset).[f] |> filterBy filter)
            |> Seq.map (fun (index, feat) -> 
                initialEntropy - conditionalEntropy feat labels, (index, feat))
            |> Seq.maxBy fst
        if (fst best > 0.) then Some(snd best) else None

    // Most likely outcome of a feature
    let private mostLikely (f: Feature) =
        f 
        |> Map.toSeq 
        |> Seq.maxBy (fun (i, s) -> Index.length s) 
        |> fst

    // Recursively build a decision tree,
    // selecting out of the remaining features
    // which ones yields the largest entropy gain
    let rec build (dataset: TrainingSet) // full dataset
                  (filter: index) // indexes of observations in use
                  (remaining: int Set) // indexes of features usable
                  (featureSelector: int Set -> int Set)
                  (minLeaf: int) = // min elements in a leaf   
                   
        if (remaining = Set.empty) then 
            Leaf(fst dataset |> filterBy filter |> mostLikely)
        elif (Index.length filter < minLeaf) then
            Leaf(fst dataset |> filterBy filter |> mostLikely)
        else
            let candidates = featureSelector remaining
            let best = selectFeature dataset filter candidates

            match best with
            | None -> Leaf(fst dataset |> filterBy filter |> mostLikely)
            | Some(best) -> 
                let (index, feature) = best
                let remaining = remaining |> Set.remove index
                let splits = filterBy filter feature
                let likely = splits |> mostLikely // what to pick when missing value?
                Branch(index, likely,
                    splits
                    |> Map.map (fun v indices ->
                           let tree = build dataset indices remaining featureSelector minLeaf
                           tree))

    // Recursively walk down the tree,
    // to figure out how an observation
    // should be classified         
    let rec decide (tree: Tree) (obs: int []) =
        match tree with
        | Leaf(outcome) -> outcome
        | Branch(feature, mostLikely, next) ->
              let value = obs.[feature]
              if Map.containsKey value next
              then decide next.[value] obs
              else decide next.[mostLikely] obs
    
    let inline any x = id x
    
    let ID3Classifier (dataset: TrainingSet) // full dataset
                      (filter: index)
                      (minLeaf: int) = // min elements in a leaf  
        let fs = snd dataset |> Array.length
        let remaining = Set.ofList [ 0 .. (fs - 1) ]
        let tree = build dataset filter remaining any minLeaf
        decide tree, tree

    // prepare an array of observed values into a Feature.
    let prepare (obs: int seq) =
        let dict = Dictionary<int, index>()
        obs
        |> Seq.fold (fun i value ->
                if dict.ContainsKey(value)
                then dict.[value] <- i::dict.[value]
                else dict.Add(value, [i])
                (i + 1)) 0
        |> ignore
        dict |> Seq.map (fun kv -> kv.Key, List.rev kv.Value) |> Map.ofSeq

    // From a sequence of observations, for a feature,
    // construct a Map of existing values to integers (indexes),
    // an extractor function that converts an observation
    // to its mapped integer index.
    let extract (feature: 'a -> 'label option) (data: 'a seq) =
        let map =
            data 
            |> Seq.choose (fun item -> feature item)
            |> Seq.distinct
            |> Seq.mapi (fun i value -> value, i + 1)
            |> Map.ofSeq
        let extractor x = 
            let value = feature x
            match value with
            | None -> 0
            | Some(value) -> if map.ContainsKey(value) then map.[value] else 0
        map, extractor

    let private append (dict: Dictionary<int, int list>) (value:int) (index:int) =
        if dict.ContainsKey(value)
        then dict.[value] <- index::dict.[value]
        else dict.Add(value, [index])

    let private prepareLabels (labels: 'label option []) =
        let labelsMap, labelizer = labels |> extract id
        let reverseLabels = 
            labelsMap 
            |> Map.toSeq 
            |> Seq.map (fun (k, v) -> (v, k)) 
            |> Map.ofSeq
        let predictor x = reverseLabels.[x]
        // currently not returning the map,
        // but might do so later for diagnosis
        labelizer, predictor

    let private prepareFeaturizer (observations: 'a seq) (fs: ('a -> 'label option) [])=
        let featuresMap, featurizers =
            fs 
            |> Array.map (fun f -> observations |> extract f)
            |> Array.unzip
        let reverseFeatures =
            featuresMap
            |> Array.map (fun map ->
                map
                |> Map.toSeq
                |> Seq.map (fun (k, v) -> (v, k))
                |> Map.ofSeq)
        // currently not returning the map,
        // but might do so later for diagnosis
        let featurizer obs = featurizers |> Array.map (fun f -> f obs)
        (featurizer, reverseFeatures)

    // Create a function to extract labels and features
    // from a sequence of training examples (with a label and features)
    let private prepareTraining (obs: ('a * 'b) seq) (fs: int) (labelizer: 'a -> int) (featurizers: ('b -> int [])) =
        let converters (l, ex) =
            labelizer l,
            featurizers ex
        
        let labels = Dictionary<int, index>()
        let features = [| for i in 1 .. fs -> (Dictionary<int, index>()) |]
        obs
        |> Seq.map (fun x -> converters x)
        |> Seq.iteri (fun i (label, feats) ->
            append labels label i
            features |> Array.iteri (fun j f -> append f feats.[j] i))

        labels |> Seq.map (fun kv -> kv.Key, List.rev kv.Value) |> Map.ofSeq,
        [| for feat in features -> feat |> Seq.map (fun kv -> kv.Key, List.rev kv.Value) |> Map.ofSeq |]

    // Tree rendering utility: renders "pipes"
    // for active branches
    let private pad (actives: int Set) (depth: int) =
        String.Join("", 
            [|  for i in 0 .. (depth - 1) do 
                    if (actives.Contains i) 
                    then yield "│   " 
                    else yield "   " |])

    // Recursively draw the nodes of the tree
    let rec private plot (tree: Tree) 
                         (actives: int Set) 
                         (depth: int) 
                         (predictor: int -> 'label)
                         (reverseFeatures: Map<int,'feature>[]) =
        match tree with
        | Leaf(x) -> printfn "%s -> %A" (pad actives depth) (predictor x)
        | Branch(f,d,next) ->        
            let last = next |> Map.toArray |> Array.length
            let fMap = reverseFeatures.[f]
            next 
            |> Map.toArray
            |> Array.iteri (fun i (x, n) -> 
                let actives' = 
                    if (i = (last - 1)) 
                    then Set.remove depth actives 
                    else actives
                let pipe = 
                    if (i = (last - 1)) 
                    then "└" else "├"
                match n with
                | Leaf(z) -> 
                    printfn "%s%s Feat %i = %A → %A" (pad actives depth) pipe f (fMap.[x]) (predictor z)
                | Branch(_) -> 
                    printfn "%s%s Feat %i = %A" (pad actives' depth) pipe f (fMap.[x]) 
                    plot n (Set.add (depth + 1) actives') (depth + 1) predictor reverseFeatures)

    // Draw the entire tree
    let render tree predictor reverseFeatures = 
        plot tree (Set.ofList [0]) 0 predictor reverseFeatures

    // Create a full ID3 Classification Tree
    let createID3Classifier (examples: ('label option * 'a) []) 
                            (fs: ('a -> 'feature option)[]) 
                            (minLeaf: int) =
        // Unwrap labels and observations
        let labels, observations = Array.unzip examples
        // Convert label to integer, and integer to label
        let labelizer, predicted = prepareLabels labels
        // Convert observation to integer array
        let featurizer, reverseMap = prepareFeaturizer observations fs
        
        let trainingSet = prepareTraining examples (Array.length fs) labelizer featurizer

        let classifier, tree = ID3Classifier trainingSet [ 0.. (examples |> Array.length) - 1 ] minLeaf
        // this needs to be returned as function result, too
        let display = render tree predicted reverseMap

        let f obs = (classifier (featurizer obs)) |> predicted
        f       

    // work in progress: Random Forest
    
    // Pick n distinct random indexes at most from a set;
    // incorrect but good enough for now.
    let pickN n (rng: Random) (from: int Set) =
        let array = Set.toArray from 
        seq { for i in 1 .. n -> array.[rng.Next(0, Array.length array)] } |> Set.ofSeq
    
    // pick a proportion p from original sample, with replacement
    let bag (p: float) (rng: Random) (from: index) =
        let size = Index.length from
        let bagSize = ((float)size * p) |> (int)
        [ for i in 1 .. bagSize -> rng.Next(0, size) ] |> List.sort

    // grow a tree, picking a random subset of the features at each node
    let private randomTree (dataset: TrainingSet) // full dataset
                           (filter: index) // indexes of observations in use
                           (remaining: int Set) // indexes of features usable
                           (minLeaf: int) 
                           (rng: Random) = // min elements in a leaf    
        let n = sqrt (Set.count remaining |> (float)) |> ceil |> (int)
        build dataset filter remaining (pickN n rng) minLeaf

    // grow a forest of random trees
    let forest (dataset: TrainingSet) // full dataset
               (filter: index) // indexes of observations in use
               (minLeaf: int) // min elements in a leaf
               (bagging: float)
               (iters: int) 
               (rng: Random) =    
        let fs = snd dataset |> Array.length
        let remaining = Set.ofList [ 0 .. (fs - 1) ]
        let n = sqrt (Set.count remaining |> (float)) |> (int)

        [| for i in 1 .. iters -> 
            let rng = Random(rng.Next())
            let picker = pickN n rng
            let bagger = bag bagging rng
            (picker, bagger) |] 
        |> Array.Parallel.map (fun (picker, bagger) -> build dataset (filter |> bagger) remaining picker minLeaf)

    // decide based on forest majority decision
    let private forestDecision (trees: Tree []) (obs: int []) =
        trees 
        |> Array.map (fun t -> decide t obs)
        |> Seq.countBy id
        |> Seq.maxBy snd
        |> fst

    let private forestDecide (trees: Tree []) (obs: 'a) (f: 'a -> int []) (l: int -> 'label) =
        forestDecision trees (f obs) |> l

    // there is obvious duplication here with ID3, need to clean up
    let createForestClassifier (examples: ('label option * 'a) []) 
                               (fs: ('a -> 'feature option)[]) 
                               (minLeaf: int) 
                               (bagging: float)
                               (iters: int) 
                               (rng: Random) =
        // Unwrap labels and observations
        let labels, observations = Array.unzip examples
        // Convert label to integer, and integer to label
        let labelizer, predicted = prepareLabels labels
        // Convert observation to integer array
        let featurizer, reverseMap = prepareFeaturizer observations fs
        
        let trainingSet = prepareTraining examples (Array.length fs) labelizer featurizer

        let forest = forest trainingSet [ 0.. (examples |> Array.length) - 1 ] minLeaf bagging iters rng
        
        let classifier obs = (forestDecide forest obs featurizer predicted)
        classifier

