#r @"C:\Gustavo\Forks\FSharp.Data\bin\FSharp.Data.dll"
#load "Index.fs"
#load "DecisionTree.fs"

open Charon
open Charon.DecisionTree
open System
open FSharp.Data

// let's define a type for our observations
type Passenger = {
    Id: int
    Class: int option
    Name: string option
    Sex: string option
    Age: decimal option
    SiblingsOrSpouse: int option
    ParentsOrChildren: int option
    Ticket: string option
    Fare: decimal option
    Cabin: string option
    Embarked: string option }

// now let's retrieve examples from the training CSV file
// http://www.kaggle.com/c/titanic-gettingStarted/data
let data = new CsvProvider<"train.csv",
                           (* PassengerId is always present, but any one of the other columns might be missing*)
                           Schema="PassengerId=int", SafeMode=true, PreferOptionals=true>() 

let trainingSet =
    [| for line in data.Data -> 
        line.Survived, // the label
        {   Id = line.PassengerId 
            Class = line.Pclass
            Name = line.Name
            Sex = line.Sex
            Age = line.Age
            SiblingsOrSpouse = line.SibSp
            ParentsOrChildren = line.Parch
            Ticket = line.Ticket
            Fare = line.Fare
            Cabin = line.Cabin
            Embarked = line.Embarked } |]

// ID3 Decision Tree example
let treeExample =
    
    // let's define what features we want included
    let features = 
        [| fun x -> x.Sex |> To.Feature
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

    let features = 
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
