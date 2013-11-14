﻿namespace Charon.Tests

module ``Experimental`` =

    open System
    open NUnit.Framework
    open FsUnit

    open Charon.Entropy
    open Charon.MDL
    open Charon.Tree

    [<Test>]
    let ``try out continuous tree`` () =

        let size = 100
        let classes = 3

        let rng = Random()

        let outcomes = [| for i in 1 .. size -> rng.Next(classes) |]

        let features = 
            [|  yield outcomes |> Array.map (fun x -> (if x = 0 then rng.NextDouble() else rng.NextDouble() + 1.), x);
                yield outcomes |> Array.map (fun x -> (if x = 2 then rng.NextDouble() else rng.NextDouble() + 1.), x);
                yield outcomes |> Array.map (fun x -> rng.NextDouble(), x); |]

        let dataset = classes, outcomes, features
        let filter = [| 0 .. (size - 1) |]
        let remaining = [0;1;2] |> Set.ofList
        let selector = id

        let tree = growTree dataset filter remaining selector 5

        42 |> should equal 42