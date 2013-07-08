#r @"C:\Gustavo\Forks\FSharp.Data\bin\FSharp.Data.dll"
#load "Index.fs"
#load "DecisionTree.fs"

open Charon
open Charon.DecisionTree
open System
open FSharp.Data

// http://www.kaggle.com/c/titanic-gettingStarted/data
type DataSet = CsvProvider<"train.csv",
                           (* PassengerId is always present, but any one of the other columns might be missing*)
                           Schema="int,,Class,,,,SiblingsOrSpouse,ParentsOrChildren", SafeMode=true, PreferOptionals=true>

type DataSet2 = CsvProvider<"train.csv",
                            (* PassengerId is always present, but any one of the other columns might be missing*)
                            Schema="PassengerId=int,Pclass->Class,Parch->ParentsOrChildren,SibSp->SiblingsOrSpouse", SafeMode=true, PreferOptionals=true>

type Passenger = DataSet.Row

let trainingSet =
    use data = new DataSet()
    [| for passenger in data.Data -> passenger.Survived, // the label
                                     passenger |]

// ID3 Decision Tree example
let treeExample =
    
    // let's define what features we want included
    let features = 
        [| fun (x:Passenger) -> x.Sex |> To.Feature
           fun x -> x.Class |> To.Feature |]

    // train the classifier
    let minLeaf = 5
    let classifier = createID3Classifier trainingSet features minLeaf

    // let's see how good the model is on the training set
    printfn "Forecast evaluation"
    let correct = 
        trainingSet
        |> Array.averageBy (fun (label, obs) -> 
            if label = Some(classifier obs) then 1. else 0.)
    printfn "Correct: %.4f" correct

// Random Forest example
let forestExample = 

    // let's define what features we want included
    let binnedAge age =
        if age < 10M
        then "Kid"
        else "Adult"

    let features : (Passenger -> _) [] = 
        [| fun x -> x.Sex |> To.Feature
           fun x -> x.Class |> To.Feature
           fun x -> x.Age |> Option.map binnedAge |> To.Feature
           fun x -> x.SiblingsOrSpouse |> To.Feature
           fun x -> x.ParentsOrChildren |> To.Feature
           fun x -> x.Embarked |> To.Feature |]

    let minLeaf = 5 // min observations per leaf
    let bagging = 0.75 // proportion of sample used for estimation
    let iters = 50 // number of trees to grow
    let rng = Random(42) // random number generator
    let forest = createForestClassifier trainingSet features minLeaf bagging iters rng
            
    let correct = 
        trainingSet
        |> Array.averageBy (fun (label, obs) -> 
            if label = Some(forest obs) then 1. else 0.)
    printfn "Correct: %.4f" correct
